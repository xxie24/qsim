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

// CUDA tile kernel for the gate-batched runner (run_qsim_gate_batch_cuda.h):
// applies a whole batch of fused gates to each shared-memory tile of the
// state. Remaps live in qubit_remap_cuda.h.

#ifndef GATE_BATCH_CUDA_KERNELS_H_
#define GATE_BATCH_CUDA_KERNELS_H_

#include <cstdint>

#include "qubit_remap_cuda.h"

namespace qsim {
namespace gate_batch_cuda {

using remap_cuda::kLanes;
using remap_cuda::RealFloatIndex;

// Largest fused gate the tile kernel applies.
constexpr unsigned kMaxGateQubits = 5;

// One executable gate of a batch, on tile-local qubits in ascending order.
// Its matrix is 2^n x 2^n complex, row-major, stored as (real, imaginary)
// float pairs at matrix_offset in the batch's matrix buffer.
struct DeviceGate {
  unsigned num_qubits;
  unsigned qubits[kMaxGateQubits];
  unsigned matrix_offset;
};

// The tile index of a group's first amplitude: `group` with a zero bit
// inserted at every gate qubit, lowest first.
template <unsigned K>
__device__ __forceinline__ unsigned GroupBaseIndex(unsigned group,
                                                   const unsigned* qubits) {
  auto base = group;
  for (unsigned q = 0; q < K; ++q) {
    const auto bit = qubits[q];
    base = ((base >> bit) << (bit + 1)) | (base & ((1u << bit) - 1));
  }
  return base;
}

// The 2^K amplitudes of one group, held in registers, with the tile index
// each came from.
template <unsigned K, typename FP>
struct AmplitudeGroup {
  static constexpr unsigned kSize = 1u << K;

  // Loads the group whose first amplitude is at `base`. Bit q of j selects
  // gate qubit qubits[q].
  __device__ __forceinline__ void Gather(const FP* tile_re, const FP* tile_im,
                                         unsigned base,
                                         const unsigned* qubits) {
#pragma unroll
    for (unsigned j = 0; j < kSize; ++j) {
      auto x = base;
#pragma unroll
      for (unsigned q = 0; q < K; ++q) {
        if ((j >> q) & 1) x |= 1u << qubits[q];
      }
      index[j] = x;
      re[j] = tile_re[x];
      im[j] = tile_im[x];
    }
  }

  // Complex matrix-vector product m * (re, im), written back to the tile at
  // the gathered indices. m is kSize x kSize, row-major, as (real,
  // imaginary) pairs.
  __device__ __forceinline__ void MultiplyAndScatter(const FP* m,
                                                     FP* tile_re,
                                                     FP* tile_im) const {
#pragma unroll
    for (unsigned r = 0; r < kSize; ++r) {
      FP rn = 0;
      FP in = 0;
#pragma unroll
      for (unsigned c = 0; c < kSize; ++c) {
        const FP mr = m[2 * (r * kSize + c)];
        const FP mi = m[2 * (r * kSize + c) + 1];
        rn += re[c] * mr - im[c] * mi;
        in += re[c] * mi + im[c] * mr;
      }
      tile_re[index[r]] = rn;
      tile_im[index[r]] = in;
    }
  }

  unsigned index[kSize];
  FP re[kSize];
  FP im[kSize];
};

// Applies one K-qubit gate to a tile held in shared memory as separate real
// and imaginary arrays. Each thread handles 2^K-amplitude groups: the
// amplitudes that differ only in the gate's qubits.
template <unsigned K, typename FP>
__device__ void ApplyGateToTile(FP* re, FP* im, unsigned tile_qubits,
                                const unsigned* qubits, const FP* m) {
  const unsigned num_groups = 1u << (tile_qubits - K);

  for (unsigned group = threadIdx.x; group < num_groups;
       group += blockDim.x) {
    AmplitudeGroup<K, FP> amplitudes;
    amplitudes.Gather(re, im, GroupBaseIndex<K>(group, qubits), qubits);
    amplitudes.MultiplyAndScatter(m, re, im);
  }
}

// Applies a zero-qubit gate: multiplies every amplitude by the phase m[0].
template <typename FP>
__device__ void ApplyPhaseToTile(FP* re, FP* im, unsigned tile_qubits,
                                 const FP* m) {
  for (unsigned i = threadIdx.x; i < (1u << tile_qubits); i += blockDim.x) {
    const FP r = re[i];
    const FP x = im[i];
    re[i] = r * m[0] - x * m[1];
    im[i] = r * m[1] + x * m[0];
  }
}

// One thread block per tile of 2^tile_qubits consecutive amplitudes: load
// the tile into shared memory, apply every gate of the batch to it, and
// write it back once. Dynamic shared memory holds the tile plus the matrix
// of the gate being applied.
template <typename FP>
__global__ void ApplyBatchToTilesKernel(FP* state, unsigned tile_qubits,
                                        const DeviceGate* gates,
                                        unsigned num_gates,
                                        const FP* matrices) {
  extern __shared__ unsigned char shared_bytes[];
  const unsigned tile_size = 1u << tile_qubits;
  FP* re = reinterpret_cast<FP*>(shared_bytes);
  FP* im = re + tile_size;
  FP* m = im + tile_size;

  FP* tile = state + uint64_t(blockIdx.x) * 2 * tile_size;
  for (unsigned i = threadIdx.x; i < tile_size; i += blockDim.x) {
    const auto f = RealFloatIndex(i);
    re[i] = tile[f];
    im[i] = tile[f + kLanes];
  }

  for (unsigned g = 0; g < num_gates; ++g) {
    const DeviceGate gate = gates[g];
    const unsigned h = 1u << gate.num_qubits;
    for (unsigned i = threadIdx.x; i < 2 * h * h; i += blockDim.x) {
      m[i] = matrices[gate.matrix_offset + i];
    }
    __syncthreads();  // Tile and matrix are ready; the last gate is done.

    switch (gate.num_qubits) {
      case 0: ApplyPhaseToTile(re, im, tile_qubits, m); break;
      case 1: ApplyGateToTile<1>(re, im, tile_qubits, gate.qubits, m); break;
      case 2: ApplyGateToTile<2>(re, im, tile_qubits, gate.qubits, m); break;
      case 3: ApplyGateToTile<3>(re, im, tile_qubits, gate.qubits, m); break;
      case 4: ApplyGateToTile<4>(re, im, tile_qubits, gate.qubits, m); break;
      case 5: ApplyGateToTile<5>(re, im, tile_qubits, gate.qubits, m); break;
    }
    __syncthreads();  // Every group is written before the next gate reads.
  }

  for (unsigned i = threadIdx.x; i < tile_size; i += blockDim.x) {
    const auto f = RealFloatIndex(i);
    tile[f] = re[i];
    tile[f + kLanes] = im[i];
  }
}

}  // namespace gate_batch_cuda
}  // namespace qsim

#endif  // GATE_BATCH_CUDA_KERNELS_H_
