// Copyright 2026 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// CUDA backend of the gate-batched runner (gate_batch_runner.h), and the
// QSimGateBatchRunnerCUDA entry point built on it (prototype).
//
// qsim's standard CUDA simulator launches one kernel per fused gate, and
// every launch reads and writes the whole state. Here a whole gate batch runs
// as one kernel: each thread block loads a tile of 2^tile_qubits amplitudes
// into shared memory, applies every fused gate of the batch to it, and writes
// it back once. Remaps are one coalesced swap pass each (qubit_remap_cuda.h),
// including swaps that move lane bits, so the planner may remap every tile
// position.
//
// Limits: float states, fused gates of at most 5 qubits. tile_qubits is an
// upper bound, lowered to what one thread block's shared memory holds (12
// qubits on sm_75, 13 on sm_86 and sm_89).

#ifndef RUN_QSIM_GATE_BATCH_CUDA_H_
#define RUN_QSIM_GATE_BATCH_CUDA_H_

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gate_batch_cuda_kernels.h"
#include "gate_batch_runner.h"
#include "qubit_remap_cuda.h"
#include "statespace_cuda.h"
#include "util_cuda.h"

namespace qsim {

template <typename IO, typename FP = float>
class CudaGateBatchBackend {
 public:
  using StateSpace = StateSpaceCUDA<FP>;
  using State = typename StateSpace::State;
  using fp_type = FP;
  using ExecutableGate = gate_batch_internal::ExecutableGate<FP>;

  struct Parameter {
    unsigned threads_per_block = 256;

    // Low positions the planner never remaps. Swaps that move lane bits run
    // through PermuteLaneGroupsKernel, which stays coalesced, so the default
    // frees all of them: fewer batches for the same swap cost.
    unsigned lane_qubits = 0;
  };

  // A tile of 2^tile_qubits amplitudes lives in shared memory (8 bytes per
  // amplitude): 13 uses 64 KiB, plus 8 KiB for the gate matrix, and was the
  // fastest on every tested circuit (RTX 4070).
  static constexpr unsigned kDefaultTileQubits = 13;
  static constexpr unsigned kDefaultEvictionFloor = 0;

  // The widest gate ApplyBatchToTilesKernel applies.
  static constexpr unsigned kMaxGateQubits = gate_batch_cuda::kMaxGateQubits;

  template <typename RunnerParameter>
  CudaGateBatchBackend(const RunnerParameter& param, unsigned num_qubits,
                       State& state)
      : param_(param),
        num_qubits_(num_qubits),
        requested_tile_qubits_(std::min(param.tile_qubits, num_qubits)),
        tile_qubits_(FitTileToSharedMemory(requested_tile_qubits_)),
        verbosity_(param.verbosity),
        state_(state.get()) {}

  CudaGateBatchBackend(const CudaGateBatchBackend&) = delete;
  CudaGateBatchBackend& operator=(const CudaGateBatchBackend&) = delete;

  ~CudaGateBatchBackend() {
    cudaFree(d_gates_);
    cudaFree(d_matrices_);
  }

  unsigned TileQubits() const { return tile_qubits_; }

  // Shrinking to fit shared memory is not a small-state signal, so the
  // planner sees the fitted size as the full request.
  unsigned RequestedTileQubits() const { return tile_qubits_; }
  unsigned LaneQubits() const { return param_.lane_qubits; }

  bool Prepare() const {
    if (param_.lane_qubits > kLayoutLaneQubits) {
      IO::errorf("qsim_gate_batch: lane_qubits must be at most %u.\n",
                 kLayoutLaneQubits);
      return false;
    }
    if (num_qubits_ < kLayoutLaneQubits || tile_qubits_ < kLayoutLaneQubits) {
      IO::errorf("qsim_gate_batch: tile_qubits and the circuit need at "
                 "least %u qubits.\n", kLayoutLaneQubits);
      return false;
    }
    LogAdaptiveTileSize();
    return true;
  }

  void ApplySwaps(const std::vector<QubitSwap>& swaps) {
    ApplyBitPairSwapsCUDA(state_, num_qubits_, swaps);
  }

  // Copies the batch's gates to the device and runs it on every tile.
  void ExecuteBatch(const std::vector<ExecutableGate>& gates) {
    host_gates_.clear();
    host_matrices_.clear();
    for (const auto& gate : gates) {
      gate_batch_cuda::DeviceGate device_gate{};
      device_gate.num_qubits = gate.physical_qubits.size();
      std::copy(gate.physical_qubits.begin(), gate.physical_qubits.end(),
                device_gate.qubits);
      device_gate.matrix_offset = host_matrices_.size();
      host_matrices_.insert(host_matrices_.end(), gate.matrix.begin(),
                            gate.matrix.end());
      host_gates_.push_back(device_gate);
    }
    Upload(host_gates_, d_gates_, gates_capacity_);
    Upload(host_matrices_, d_matrices_, matrices_capacity_);

    const unsigned num_tiles = 1u << (num_qubits_ - tile_qubits_);
    auto kernel = gate_batch_cuda::ApplyBatchToTilesKernel<FP>;
    ErrorCheck(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        TileSharedBytes(tile_qubits_)));
    kernel<<<num_tiles, param_.threads_per_block,
             TileSharedBytes(tile_qubits_)>>>(
        state_, tile_qubits_, d_gates_, unsigned(host_gates_.size()),
        d_matrices_);
    ErrorCheck(cudaGetLastError());
  }

  void Synchronize() const { ErrorCheck(cudaDeviceSynchronize()); }

 private:
  // Tiles need at least one lane group of StateSpaceCUDA (one warp).
  static constexpr unsigned kLayoutLaneQubits = remap_cuda::kLaneQubits;

  // Tile (real and imaginary arrays) plus the largest gate matrix.
  static std::size_t TileSharedBytes(unsigned tile_qubits) {
    const std::size_t h = std::size_t{1} << gate_batch_cuda::kMaxGateQubits;
    return sizeof(FP) * (2 * (std::size_t{1} << tile_qubits) + 2 * h * h);
  }

  static std::size_t MaxSharedBytesPerBlock() {
    int device = 0;
    int bytes = 0;
    ErrorCheck(cudaGetDevice(&device));
    ErrorCheck(cudaDeviceGetAttribute(
        &bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return bytes;
  }

  // The largest tile, up to the requested size, that fits one thread block's
  // shared memory on the current device.
  static unsigned FitTileToSharedMemory(unsigned tile_qubits) {
    const auto max_bytes = MaxSharedBytesPerBlock();
    while (tile_qubits > kLayoutLaneQubits &&
           TileSharedBytes(tile_qubits) > max_bytes) {
      --tile_qubits;
    }
    return tile_qubits;
  }

  void LogAdaptiveTileSize() const {
    if (verbosity_ > 1 && tile_qubits_ != requested_tile_qubits_) {
      IO::messagef("adaptive tile size: L=%u reduced to L=%u to fit %zu "
                   "bytes of shared memory per block.\n",
                   requested_tile_qubits_, tile_qubits_,
                   MaxSharedBytesPerBlock());
    }
  }

  template <typename T>
  static void Upload(const std::vector<T>& host, T*& device,
                     std::size_t& capacity) {
    if (host.size() > capacity) {
      cudaFree(device);
      capacity = 2 * host.size();
      ErrorCheck(cudaMalloc(&device, capacity * sizeof(T)));
    }
    ErrorCheck(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  const Parameter& param_;
  const unsigned num_qubits_;
  const unsigned requested_tile_qubits_;
  const unsigned tile_qubits_;
  const unsigned verbosity_;
  FP* const state_;
  std::vector<gate_batch_cuda::DeviceGate> host_gates_;
  std::vector<FP> host_matrices_;
  gate_batch_cuda::DeviceGate* d_gates_ = nullptr;
  FP* d_matrices_ = nullptr;
  std::size_t gates_capacity_ = 0;
  std::size_t matrices_capacity_ = 0;
};

// The CUDA gate-batched runner: the same planner and pipeline as
// QSimGateBatchRunner, with batches executed on the GPU.
template <typename IO, typename Fuser, typename FP = float>
using QSimGateBatchRunnerCUDA =
    GateBatchRunner<IO, Fuser, CudaGateBatchBackend<IO, FP>>;

}  // namespace qsim

#endif  // RUN_QSIM_GATE_BATCH_CUDA_H_
