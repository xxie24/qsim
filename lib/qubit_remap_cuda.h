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

// Qubit remapping for StateSpaceCUDA states: the CUDA counterpart of
// qubit_remap.h.
//
// Applies a set of DISJOINT amplitude-bit transpositions to the state, in
// place, in one pass. StateSpaceCUDA stores lane groups of 32 amplitudes
// (one warp): 32 real parts followed by 32 imaginary parts, so amplitude a
// has its real part at float 64 * (a / 32) + a % 32 and its imaginary part
// 32 floats later. The low 5 amplitude bits select the lane.
//
// Swaps above the lane bits move whole lane groups (SwapBitPairsKernel), which
// keeps every warp access coalesced. Unlike the CPU version, swaps may also
// move lane bits: PermuteLaneGroupsKernel loads whole lane groups into shared
// memory, permutes them there, and stores them whole, so global memory still
// sees only coalesced accesses.

#ifndef QUBIT_REMAP_CUDA_H_
#define QUBIT_REMAP_CUDA_H_

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "qubit_layout.h"
#include "util_cuda.h"

namespace qsim {
namespace remap_cuda {

// Amplitude-index bits inside one lane group of StateSpaceCUDA (one warp).
constexpr unsigned kLaneQubits = 5;
constexpr unsigned kLanes = 1u << kLaneQubits;  // Amplitudes per lane group.
constexpr unsigned kGroupFloats = 2 * kLanes;   // kLanes reals, kLanes imags.

// A disjoint set of amplitude-bit transpositions; at most 32 fit a 64-qubit
// index.
constexpr unsigned kMaxSwapPairs = 32;

struct SwapPairs {
  unsigned num_pairs;
  unsigned lower[kMaxSwapPairs];
  unsigned upper[kMaxSwapPairs];
};

// Float offset of an amplitude's real part; its imaginary part is kLanes
// floats later.
__host__ __device__ __forceinline__ uint64_t RealFloatIndex(
    uint64_t amplitude) {
  return kGroupFloats * (amplitude >> kLaneQubits) +
         (amplitude & (kLanes - 1));
}

// Applies every transposition in `pairs` in one pass. Each moved amplitude
// pair is swapped by the thread that owns the smaller index.
template <typename FP>
__global__ void SwapBitPairsKernel(FP* state, uint64_t num_amplitudes,
                                   SwapPairs pairs) {
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t a = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       a < num_amplitudes; a += stride) {
    auto b = a;
    for (unsigned p = 0; p < pairs.num_pairs; ++p) {
      const auto lower = pairs.lower[p];
      const auto upper = pairs.upper[p];
      if (((a >> lower) ^ (a >> upper)) & 1) {
        b ^= (uint64_t{1} << lower) | (uint64_t{1} << upper);
      }
    }
    if (b > a) {
      const auto fa = RealFloatIndex(a);
      const auto fb = RealFloatIndex(b);
      const FP re = state[fa];
      const FP im = state[fa + kLanes];
      state[fa] = state[fb];
      state[fa + kLanes] = state[fb + kLanes];
      state[fb] = re;
      state[fb + kLanes] = im;
    }
  }
}

// A swap set whose pairs may move lane bits (positions below 5). A pair
// (lower < 5 <= upper) is a cross pair: it trades lanes between lane groups
// that differ in group-index bit upper - 5.
struct LaneGroupPermutation {
  SwapPairs pairs;
  unsigned num_cross;                // Cross pairs, at most kLaneQubits.
  unsigned cross_bits[kLaneQubits];  // Their group-index bits
                                     // (upper - kLaneQubits), ascending.
};

__device__ __forceinline__ uint64_t SwapAmplitudeBits(uint64_t a,
                                                      const SwapPairs& p) {
  auto b = a;
  for (unsigned i = 0; i < p.num_pairs; ++i) {
    if (((a >> p.lower[i]) ^ (a >> p.upper[i])) & 1) {
      b ^= (uint64_t{1} << p.lower[i]) | (uint64_t{1} << p.upper[i]);
    }
  }
  return b;
}

// Applies any disjoint swap set, lane bits included, while reading and
// writing global memory only in whole 32-amplitude lane groups. Lane groups
// are processed in closed units: a base group b (all cross bits zero), the
// 2^num_cross groups that differ from it in cross bits, and the same set
// around its partner P(b) under the high-bit swaps. Each unit is loaded
// whole into shared memory, permuted there, and stored whole, so the
// scattering happens on chip. The unit with the smaller base does the work.
template <typename FP>
__global__ void PermuteLaneGroupsKernel(FP* state, uint64_t num_units,
                                        unsigned group_bits,
                                        LaneGroupPermutation perm) {
  extern __shared__ unsigned char shared_bytes[];
  const unsigned k = perm.num_cross;
  const unsigned half = 1u << k;  // Groups per bundle.
  FP* in = reinterpret_cast<FP*>(shared_bytes);
  FP* out = in + 2 * half * kGroupFloats;

  uint64_t cross_mask = 0;
  for (unsigned i = 0; i < k; ++i) {
    cross_mask |= uint64_t{1} << perm.cross_bits[i];
  }

  for (uint64_t unit = blockIdx.x; unit < num_units; unit += gridDim.x) {
    // Base group: deposit the unit index into the non-cross group bits.
    uint64_t base = 0;
    for (unsigned bit = 0, j = 0; bit < group_bits; ++bit) {
      if ((cross_mask >> bit) & 1) continue;
      base |= ((unit >> j) & 1) << bit;
      ++j;
    }
    const uint64_t partner =
        SwapAmplitudeBits(base << kLaneQubits, perm.pairs) >> kLaneQubits;
    if (partner < base) continue;  // The partner's unit does the work.
    const unsigned num_groups = partner == base ? half : 2 * half;

    // Slot s holds bundle (s >= half) group (s % half): base or partner,
    // with cross bits taken from the bits of s % half.
    auto group_of = [&](unsigned slot) {
      uint64_t g = slot < half ? base : partner;
      const unsigned i = slot % half;
      for (unsigned c = 0; c < k; ++c) {
        if ((i >> c) & 1) g |= uint64_t{1} << perm.cross_bits[c];
      }
      return g;
    };
    auto slot_of = [&](uint64_t group) {
      unsigned i = 0;
      for (unsigned c = 0; c < k; ++c) {
        i |= unsigned((group >> perm.cross_bits[c]) & 1) << c;
      }
      return ((group & ~cross_mask) == base ? 0u : half) + i;
    };

    for (unsigned f = threadIdx.x; f < num_groups * kGroupFloats;
         f += blockDim.x) {
      in[f] = state[group_of(f / kGroupFloats) * kGroupFloats +
                    f % kGroupFloats];
    }
    __syncthreads();

    for (unsigned t = threadIdx.x; t < num_groups * kLanes; t += blockDim.x) {
      const unsigned slot = t / kLanes;
      const unsigned lane = t % kLanes;
      const uint64_t dest = (group_of(slot) << kLaneQubits) | lane;
      const uint64_t src = SwapAmplitudeBits(dest, perm.pairs);
      const unsigned from = slot_of(src >> kLaneQubits) * kGroupFloats +
                            unsigned(src & (kLanes - 1));
      out[slot * kGroupFloats + lane] = in[from];
      out[slot * kGroupFloats + lane + kLanes] = in[from + kLanes];
    }
    __syncthreads();

    for (unsigned f = threadIdx.x; f < num_groups * kGroupFloats;
         f += blockDim.x) {
      state[group_of(f / kGroupFloats) * kGroupFloats + f % kGroupFloats] =
          out[f];
    }
    __syncthreads();  // Shared buffers are reused by the next unit.
  }
}

// Launch shape for the grid-stride swap kernels: enough blocks to fill the
// GPU (the loops cover any remainder). Not tuned.
constexpr unsigned kThreadsPerBlock = 256;
constexpr unsigned kSwapBlocks = 4096;
constexpr unsigned kMaxPermuteBlocks = 1u << 18;

}  // namespace remap_cuda

// Applies all `qubit_swaps` transpositions of amplitude-bit positions to a
// StateSpaceCUDA state, in place, in a single pass. The pairs must be
// disjoint. The launch is asynchronous.
template <typename FP>
inline void ApplyBitPairSwapsCUDA(FP* state, unsigned num_qubits,
                                  const std::vector<QubitSwap>& qubit_swaps) {
  namespace rc = remap_cuda;
  if (qubit_swaps.empty()) return;

  rc::LaneGroupPermutation perm{};
  auto& pairs = perm.pairs;
  pairs.num_pairs = qubit_swaps.size();
  bool moves_lane_bits = false;
  for (std::size_t i = 0; i < qubit_swaps.size(); ++i) {
    pairs.lower[i] = std::min(qubit_swaps[i].first, qubit_swaps[i].second);
    pairs.upper[i] = std::max(qubit_swaps[i].first, qubit_swaps[i].second);
    if (pairs.lower[i] < rc::kLaneQubits) moves_lane_bits = true;
    if (pairs.lower[i] < rc::kLaneQubits && pairs.upper[i] >= rc::kLaneQubits) {
      perm.cross_bits[perm.num_cross++] = pairs.upper[i] - rc::kLaneQubits;
    }
  }
  std::sort(perm.cross_bits, perm.cross_bits + perm.num_cross);

  if (!moves_lane_bits) {
    // Whole lane groups trade places: already coalesced.
    const uint64_t num_amplitudes = uint64_t{1} << num_qubits;
    rc::SwapBitPairsKernel<FP><<<rc::kSwapBlocks, rc::kThreadsPerBlock>>>(
        state, num_amplitudes, pairs);
  } else {
    const unsigned group_bits = num_qubits - rc::kLaneQubits;
    const uint64_t num_units = uint64_t{1} << (group_bits - perm.num_cross);
    const unsigned max_groups = 2u << perm.num_cross;
    const auto threads =
        std::min(rc::kThreadsPerBlock, max_groups * rc::kLanes);
    const auto blocks =
        unsigned(std::min<uint64_t>(num_units, rc::kMaxPermuteBlocks));
    const std::size_t shared =
        2 * max_groups * rc::kGroupFloats * sizeof(FP);
    rc::PermuteLaneGroupsKernel<FP><<<blocks, threads, shared>>>(
        state, num_units, group_bits, perm);
  }
  ErrorCheck(cudaGetLastError());
}

}  // namespace qsim

#endif  // QUBIT_REMAP_CUDA_H_
