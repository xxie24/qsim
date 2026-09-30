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
  static constexpr unsigned kDefaultEvictionFloor = 5;

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
    unsigned max_gate_qubits = 0;
    for (const auto& gate : gates) {
      const unsigned k = gate.physical_qubits.size();
      max_gate_qubits = std::max(max_gate_qubits, k);
      const unsigned offset = host_matrices_.size();
      host_matrices_.insert(host_matrices_.end(), gate.matrix.begin(),
                            gate.matrix.end());
      host_gates_.push_back(BuildDeviceGate(tile_qubits_, gate, offset));
    }
    Upload(host_gates_, d_gates_, gates_capacity_);
    Upload(host_matrices_, d_matrices_, matrices_capacity_);

    const unsigned h_max = 1u << max_gate_qubits;
    const unsigned matrix_stride = 2u * h_max * h_max;
    const std::size_t shared_bytes =
        TileSharedBytes(tile_qubits_, max_gate_qubits);
    const unsigned num_tiles = 1u << (num_qubits_ - tile_qubits_);
    const unsigned k_dispatch = std::max(1u, max_gate_qubits);
    if (k_dispatch <= 3) {
      auto kernel = gate_batch_cuda::ApplyBatchToTilesKernel<FP, 3>;
      ErrorCheck(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
      kernel<<<num_tiles, param_.threads_per_block, shared_bytes>>>(
          state_, tile_qubits_, d_gates_, unsigned(host_gates_.size()),
          d_matrices_, matrix_stride);
    } else if (k_dispatch == 4) {
      auto kernel = gate_batch_cuda::ApplyBatchToTilesKernel<FP, 4>;
      ErrorCheck(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
      kernel<<<num_tiles, param_.threads_per_block, shared_bytes>>>(
          state_, tile_qubits_, d_gates_, unsigned(host_gates_.size()),
          d_matrices_, matrix_stride);
    } else {
      auto kernel = gate_batch_cuda::ApplyBatchToTilesKernel<FP, 5>;
      ErrorCheck(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
      kernel<<<num_tiles, param_.threads_per_block, shared_bytes>>>(
          state_, tile_qubits_, d_gates_, unsigned(host_gates_.size()),
          d_matrices_, matrix_stride);
    }
    ErrorCheck(cudaGetLastError());
  }

  void Synchronize() const { ErrorCheck(cudaDeviceSynchronize()); }

 private:
  // Tiles need at least one lane group of StateSpaceCUDA (one warp).
  static constexpr unsigned kLayoutLaneQubits = remap_cuda::kLaneQubits;

  static gate_batch_cuda::DeviceGate BuildDeviceGate(
      unsigned tile_qubits, const ExecutableGate& gate,
      unsigned matrix_offset) {
    using gate_batch_cuda::Swizzle;
    gate_batch_cuda::DeviceGate dg{};
    const unsigned k = gate.physical_qubits.size();
    dg.num_qubits = k;
    dg.matrix_offset = matrix_offset;
    dg.w_clear_mask = ~31u;
    if (k == 0) return dg;

    unsigned ng[64];
    unsigned num_g_bits = 0;
    for (unsigned p = 0, qi = 0; p < tile_qubits; ++p) {
      if (qi < k && gate.physical_qubits[qi] == p) {
        ++qi;
      } else {
        ng[num_g_bits++] = p;
      }
    }

    const unsigned low_bits = std::min(kLayoutLaneQubits, num_g_bits);
    unsigned basis[kLayoutLaneQubits] = {};
    unsigned basis_size = 0;
    unsigned dep_low[kLayoutLaneQubits] = {};
    unsigned num_dep = 0;
    for (unsigned b = 0; b < low_bits; ++b) {
      unsigned v = Swizzle(1u << ng[b]) & 31u;
      for (unsigned i = 0; i < basis_size; ++i) {
        v = std::min(v, v ^ basis[i]);
      }
      if (v != 0) {
        basis[basis_size++] = v;
        std::sort(basis, basis + basis_size, std::greater<unsigned>());
      } else {
        dep_low[num_dep++] = b;
      }
    }

    unsigned target[64];
    for (unsigned i = 0; i < num_g_bits; ++i) target[i] = i;
    unsigned num_swaps = 0;
    for (unsigned d = 0; d < num_dep && num_swaps < 2; ++d) {
      const unsigned b = dep_low[d];
      for (unsigned hi = kLayoutLaneQubits; hi < num_g_bits; ++hi) {
        if (target[hi] != hi) continue;
        unsigned v = Swizzle(1u << ng[hi]) & 31u;
        for (unsigned i = 0; i < basis_size; ++i) {
          v = std::min(v, v ^ basis[i]);
        }
        if (v != 0) {
          basis[basis_size++] = v;
          std::sort(basis, basis + basis_size, std::greater<unsigned>());
          target[b] = hi;
          target[hi] = b;
          dg.w_clear_mask &= ~(1u << hi);
          if (num_swaps == 0) {
            dg.w_shift0 = hi - b;
            dg.w_bit0 = 1u << b;
          } else {
            dg.w_shift1 = hi - b;
            dg.w_bit1 = 1u << b;
          }
          ++num_swaps;
          break;
        }
      }
    }

    for (unsigned b = 0; b < kLayoutLaneQubits; ++b) {
      dg.lane_swizzled[b] =
          (b < low_bits) ? Swizzle(1u << ng[target[b]]) : 0u;
    }

    const auto& qubits = gate.physical_qubits;
    dg.mss[0] = (1u << qubits[0]) - 1u;
    for (unsigned q = 1; q < k; ++q) {
      dg.mss[q] =
          ((1u << qubits[q]) - 1u) ^ ((1u << (qubits[q - 1] + 1)) - 1u);
    }
    dg.mss[k] = ~((1u << (qubits[k - 1] + 1)) - 1u);

    for (unsigned j = 0; j < (1u << k); ++j) {
      unsigned x = 0;
      for (unsigned q = 0; q < k; ++q) {
        if ((j >> q) & 1) x |= 1u << qubits[q];
      }
      dg.swizzled_xss[j] = Swizzle(x);
    }
    return dg;
  }

  // Tile (real and imaginary arrays) plus double-buffered gate matrices and
  // DeviceGate descriptors.
  static std::size_t TileSharedBytes(
      unsigned tile_qubits,
      unsigned max_gate_qubits = gate_batch_cuda::kMaxGateQubits) {
    const std::size_t h = std::size_t{1} << max_gate_qubits;
    return sizeof(FP) * (2 * (std::size_t{1} << tile_qubits) + 4 * h * h) +
           2 * sizeof(gate_batch_cuda::DeviceGate);
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
