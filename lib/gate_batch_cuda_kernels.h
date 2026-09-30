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

// Linear XOR bank swizzle on tile amplitude index: folds bits >= 5 into
// the 5 warp-lane bank bits (0..4) so threads in a warp access distinct
// shared-memory banks even when gate qubits occupy low bit positions.
// Self-inverse (Swizzle(Swizzle(z)) == z) and XOR-linear
// (Swizzle(a ^ b) == Swizzle(a) ^ Swizzle(b)).
__host__ __device__ __forceinline__ unsigned Swizzle(unsigned z) {
  return z ^ (((z >> 5) ^ (z >> 8) ^ (z >> 11)) & 31u);
}

// Precomputed gate descriptor for one executable gate of a batch on
// tile-local qubits in ascending order.
struct DeviceGate {
  unsigned num_qubits;
  unsigned matrix_offset;
  unsigned w_clear_mask;
  unsigned w_shift0;
  unsigned w_bit0;
  unsigned w_shift1;
  unsigned w_bit1;
  unsigned lane_swizzled[remap_cuda::kLaneQubits];
  unsigned mss[kMaxGateQubits + 1];
  unsigned swizzled_xss[1u << kMaxGateQubits];
};

// Computes the warp-uniform swizzled base address contribution for the high
// group bits (g_hi = group & ~31u), routing up to two displaced warp bits into
// low positions to guarantee conflict-free warp lane bank coverage.
template <unsigned K>
__device__ __forceinline__ unsigned GroupSwizzledBaseHi(
    unsigned g_hi, const DeviceGate& gate) {
  unsigned w_perm = (g_hi & gate.w_clear_mask) |
                    ((g_hi >> gate.w_shift0) & gate.w_bit0) |
                    ((g_hi >> gate.w_shift1) & gate.w_bit1);
  unsigned base_hi = w_perm & gate.mss[0];
#pragma unroll
  for (unsigned q = 1; q <= K; ++q) {
    w_perm <<= 1;
    base_hi |= w_perm & gate.mss[q];
  }
  return Swizzle(base_hi);
}

// The 2^K amplitudes of one group, held in registers without an index[] array.
template <unsigned K, typename FP>
struct AmplitudeGroup {
  static constexpr unsigned kSize = 1u << K;

  __device__ __forceinline__ void Gather(const FP* __restrict__ tile_re,
                                         const FP* __restrict__ tile_im,
                                         unsigned swizzled_base,
                                         const unsigned* __restrict__ sw_xss) {
#pragma unroll
    for (unsigned j = 0; j < kSize; ++j) {
      const unsigned x = swizzled_base ^ sw_xss[j];
      re[j] = tile_re[x];
      im[j] = tile_im[x];
    }
  }

  __device__ __forceinline__ void MultiplyAndScatter(
      const FP* __restrict__ m, unsigned swizzled_base,
      const unsigned* __restrict__ sw_xss, FP* __restrict__ tile_re,
      FP* __restrict__ tile_im) const {
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
      const unsigned x = swizzled_base ^ sw_xss[r];
      tile_re[x] = rn;
      tile_im[x] = in;
    }
  }

  FP re[kSize];
  FP im[kSize];
};

// Applies one K-qubit gate to a swizzled tile held in shared memory as
// separate real and imaginary arrays.
template <unsigned K, typename FP>
__device__ __forceinline__ void ApplyGateToTile(
    FP* __restrict__ re, FP* __restrict__ im, unsigned tile_qubits,
    const DeviceGate& gate, const FP* __restrict__ m) {
  const unsigned num_groups = 1u << (tile_qubits - K);
  const unsigned lane = threadIdx.x & 31u;
  unsigned my_swizzled_lane = 0;
#pragma unroll
  for (unsigned b = 0; b < remap_cuda::kLaneQubits; ++b) {
    if ((lane >> b) & 1) my_swizzled_lane ^= gate.lane_swizzled[b];
  }

  for (unsigned group = threadIdx.x; group < num_groups;
       group += blockDim.x) {
    const unsigned swizzled_base =
        GroupSwizzledBaseHi<K>(group & ~31u, gate) ^ my_swizzled_lane;
    AmplitudeGroup<K, FP> amplitudes;
    amplitudes.Gather(re, im, swizzled_base, gate.swizzled_xss);
    amplitudes.MultiplyAndScatter(m, swizzled_base, gate.swizzled_xss, re, im);
  }
}

// Applies a zero-qubit gate: multiplies every amplitude by the phase m[0].
template <typename FP>
__device__ __forceinline__ void ApplyPhaseToTile(FP* __restrict__ re,
                                                 FP* __restrict__ im,
                                                 unsigned tile_qubits,
                                                 const FP* __restrict__ m) {
  const FP mr = m[0];
  const FP mi = m[1];
  for (unsigned i = threadIdx.x; i < (1u << tile_qubits); i += blockDim.x) {
    const FP r = re[i];
    const FP x = im[i];
    re[i] = r * mr - x * mi;
    im[i] = r * mi + x * mr;
  }
}

// One thread block per tile of 2^tile_qubits consecutive amplitudes: load
// the tile into swizzled shared memory, apply every gate of the batch using
// double-buffered gate descriptors/matrices (1 barrier per gate), and write
// the tile back once.
template <typename FP, unsigned MaxK = 5>
__global__ __launch_bounds__(512) void ApplyBatchToTilesKernel(
    FP* __restrict__ state, unsigned tile_qubits,
    const DeviceGate* __restrict__ gates, unsigned num_gates,
    const FP* __restrict__ matrices, unsigned matrix_stride) {
  extern __shared__ unsigned char shared_bytes[];
  const unsigned tile_size = 1u << tile_qubits;
  FP* __restrict__ re = reinterpret_cast<FP*>(shared_bytes);
  FP* __restrict__ im = re + tile_size;
  FP* __restrict__ m_buf = im + tile_size;
  DeviceGate* __restrict__ s_gates =
      reinterpret_cast<DeviceGate*>(m_buf + 2 * matrix_stride);

  FP* tile = state + uint64_t(blockIdx.x) * 2 * tile_size;
  for (unsigned i = threadIdx.x; i < tile_size; i += blockDim.x) {
    const auto f = RealFloatIndex(i);
    const unsigned s = Swizzle(i);
    re[s] = tile[f];
    im[s] = tile[f + kLanes];
  }

  constexpr unsigned kGateWords = sizeof(DeviceGate) / sizeof(unsigned);
  if (num_gates > 0) {
    for (unsigned i = threadIdx.x; i < kGateWords; i += blockDim.x) {
      reinterpret_cast<unsigned*>(&s_gates[0])[i] =
          reinterpret_cast<const unsigned*>(&gates[0])[i];
    }
    const unsigned h0 = 1u << gates[0].num_qubits;
    const unsigned off0 = gates[0].matrix_offset;
    for (unsigned i = threadIdx.x; i < 2 * h0 * h0; i += blockDim.x) {
      m_buf[i] = matrices[off0 + i];
    }
  }
  __syncthreads();

  for (unsigned g = 0; g < num_gates; ++g) {
    const DeviceGate& gate = s_gates[g & 1];
    const FP* __restrict__ m = m_buf + (g & 1) * matrix_stride;

    if (g + 1 < num_gates) {
      const unsigned next_slot = (g + 1) & 1;
      for (unsigned i = threadIdx.x; i < kGateWords; i += blockDim.x) {
        reinterpret_cast<unsigned*>(&s_gates[next_slot])[i] =
            reinterpret_cast<const unsigned*>(&gates[g + 1])[i];
      }
      const unsigned h1 = 1u << gates[g + 1].num_qubits;
      const unsigned off1 = gates[g + 1].matrix_offset;
      FP* __restrict__ m_next = m_buf + next_slot * matrix_stride;
      for (unsigned i = threadIdx.x; i < 2 * h1 * h1; i += blockDim.x) {
        m_next[i] = matrices[off1 + i];
      }
    }

    switch (gate.num_qubits) {
      case 0: ApplyPhaseToTile(re, im, tile_qubits, m); break;
      case 1: ApplyGateToTile<1>(re, im, tile_qubits, gate, m); break;
      case 2: ApplyGateToTile<2>(re, im, tile_qubits, gate, m); break;
      case 3: ApplyGateToTile<3>(re, im, tile_qubits, gate, m); break;
      case 4:
        if constexpr (MaxK >= 4) {
          ApplyGateToTile<4>(re, im, tile_qubits, gate, m);
        }
        break;
      case 5:
        if constexpr (MaxK >= 5) {
          ApplyGateToTile<5>(re, im, tile_qubits, gate, m);
        }
        break;
    }
    __syncthreads();
  }

  for (unsigned i = threadIdx.x; i < tile_size; i += blockDim.x) {
    const auto f = RealFloatIndex(i);
    const unsigned s = Swizzle(i);
    tile[f] = re[s];
    tile[f + kLanes] = im[s];
  }
}

}  // namespace gate_batch_cuda
}  // namespace qsim

#endif  // GATE_BATCH_CUDA_KERNELS_H_
