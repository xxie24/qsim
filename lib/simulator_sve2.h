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

#ifndef SIMULATOR_SVE2_H_
#define SIMULATOR_SVE2_H_

#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#if !defined(__ARM_FEATURE_SVE2)
#error "simulator_sve2.h requires SVE2 support."
#endif
#include <arm_sve.h>

#include "simulator.h"
#include "simulator_basic.h"
#include "statespace_basic.h"
#include "statespace_sve.h"

// Macros to clean up SVE sizeless-type register unrolling
#define QSIM_SVE_COMPUTE_STEP(I)                                       \
  svfloat32_t m_kl##I = svreinterpret_f32_f64(svdup_n_f64(*p_v##I++)); \
  rn##I = svcmla_f32_x(pg32, rn##I, sv_l, m_kl##I, 0);                 \
  rn##I = svcmla_f32_x(pg32, rn##I, sv_l, m_kl##I, 90);

#define QSIM_SVE_SCATTER_STEP(I)                        \
  svst1_scatter_u64offset_f64(                          \
      pg64, (double*)(p0 + xss[k + I]), v_lane_offsets, \
      svreinterpret_f64_f32(rn##I));

#define QSIM_SVE_CTRL_ORG_STEP(I)                                 \
  svfloat32_t org##I;                                             \
  if constexpr (N <= 2) {                                         \
    org##I = svreinterpret_f32_f64(svld1_gather_u64offset_f64(    \
        pg64, (const double*)(p0 + xss[k + I]), v_lane_offsets)); \
  } else {                                                        \
    org##I = svld1_f32(pg32, tmp + (k + I) * vl);                 \
  }                                                               \
  rn##I = svsel_f32(c_match, rn##I, org##I);

namespace qsim {

template <typename For>
class SimulatorSVE2 final : public SimulatorBase {
 public:
  using StateSpace = StateSpaceSVE<For>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using BasicStateSpace = StateSpaceBasic<For, float>;
  using BasicState = typename BasicStateSpace::State;

  template <typename... ForArgs>
  explicit SimulatorSVE2(ForArgs&&... args)
      : for_(args...), basic_state_space_(args...), fallback_(args...) {}

  void ApplyGate(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      State& state) const {
    uint64_t vl64 = svcntd();
    unsigned k_param = bits::Log2(vl64) + qs.size();

    if (state.num_qubits() < k_param) {
      ApplyGateFallback(qs, matrix, state);
      return;
    }

    switch (qs.size()) {
      case 1:
        ApplyGateCore<1>(qs, matrix, state);
        break;
      case 2:
        ApplyGateCore<2>(qs, matrix, state);
        break;
      case 3:
        ApplyGateCore<3>(qs, matrix, state);
        break;
      case 4:
        ApplyGateCore<4>(qs, matrix, state);
        break;
      case 5:
        ApplyGateCore<5>(qs, matrix, state);
        break;
      case 6:
        ApplyGateCore<6>(qs, matrix, state);
        break;
      default:
        ApplyGateFallback(qs, matrix, state);
        break;
    }
  }

  void ApplyControlledGate(
      const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs,
      uint64_t cvals, const fp_type* matrix, State& state) const {
    if (cqs.empty()) {
      ApplyGate(qs, matrix, state);
      return;
    }

    uint64_t vl64 = svcntd();
    unsigned k_param = bits::Log2(vl64) + qs.size() + cqs.size();

    if (state.num_qubits() < k_param) {
      ApplyControlledGateFallback(qs, cqs, cvals, matrix, state);
      return;
    }

    switch (qs.size()) {
      case 1:
        ApplyControlledGateCore<1>(qs, cqs, cvals, matrix, state);
        break;
      case 2:
        ApplyControlledGateCore<2>(qs, cqs, cvals, matrix, state);
        break;
      case 3:
        ApplyControlledGateCore<3>(qs, cqs, cvals, matrix, state);
        break;
      case 4:
        ApplyControlledGateCore<4>(qs, cqs, cvals, matrix, state);
        break;
      case 5:
        ApplyControlledGateCore<5>(qs, cqs, cvals, matrix, state);
        break;
      case 6:
        ApplyControlledGateCore<6>(qs, cqs, cvals, matrix, state);
        break;
      default:
        ApplyControlledGateFallback(qs, cqs, cvals, matrix, state);
        break;
    }
  }

  std::complex<double> ExpectationValue(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      const State& state) const {
    uint64_t vl64 = svcntd();
    unsigned k_param = bits::Log2(vl64) + qs.size();

    if (state.num_qubits() < k_param) {
      auto basic_state = CreateBasicState(state);
      return fallback_.ExpectationValue(qs, matrix, basic_state);
    }

    switch (qs.size()) {
      case 1:
        return ExpectationValueCore<1>(qs, matrix, state);
      case 2:
        return ExpectationValueCore<2>(qs, matrix, state);
      case 3:
        return ExpectationValueCore<3>(qs, matrix, state);
      case 4:
        return ExpectationValueCore<4>(qs, matrix, state);
      case 5:
        return ExpectationValueCore<5>(qs, matrix, state);
      case 6:
        return ExpectationValueCore<6>(qs, matrix, state);
      default:
        break;
    }

    auto basic_state = CreateBasicState(state);
    return fallback_.ExpectationValue(qs, matrix, basic_state);
  }

  BasicState CreateBasicState(const State& state) const {
    auto basic_state = BasicStateSpace::Create(state.num_qubits());
    CopySveStateToBasic(state, basic_state);
    return basic_state;
  }

  void CopySveStateToBasic(const State& src, BasicState& dest) const {
    uint64_t size = uint64_t{1} << src.num_qubits();
    for (uint64_t i = 0; i < size; ++i) {
      BasicStateSpace::SetAmpl(dest, i, StateSpace::GetAmpl(src, i));
    }
  }

  void CopyBasicStateToSve(const BasicState& src, State& dest) const {
    uint64_t size = uint64_t{1} << dest.num_qubits();
    for (uint64_t i = 0; i < size; ++i) {
      StateSpace::SetAmpl(dest, i, BasicStateSpace::GetAmpl(src, i));
    }
  }

  void ApplyGateFallback(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      State& state) const {
    auto basic_state = CreateBasicState(state);
    fallback_.ApplyGate(qs, matrix, basic_state);
    CopyBasicStateToSve(basic_state, state);
  }

  void ApplyControlledGateFallback(
      const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs,
      uint64_t cvals, const fp_type* matrix, State& state) const {
    auto basic_state = CreateBasicState(state);
    fallback_.ApplyControlledGate(qs, cqs, cvals, matrix, basic_state);
    CopyBasicStateToSve(basic_state, state);
  }

  static unsigned SIMDRegisterSize() { return svcntw(); }

 private:
  template <unsigned N>
  void ApplyGateCore(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      State& state) const {
    constexpr unsigned gsize = 1 << N;
    uint64_t vl64 = svcntd();
    svbool_t pg64 = svptrue_b64();

    uint64_t ms[N + 1];
    uint64_t xss[gsize];
    FillIndices<N>(state.num_qubits(), qs, ms, xss);

    uint64_t lane_offsets_buf[64];
    for (uint64_t lane = 0; lane < vl64; ++lane) {
      uint64_t shifted = lane;
      uint64_t ii_lane = lane & ms[0];
      for (unsigned j = 1; j <= N; ++j) {
        shifted *= 2;
        ii_lane |= shifted & ms[j];
      }
      lane_offsets_buf[lane] = ii_lane * 2 * sizeof(fp_type);
    }
    svuint64_t v_lane_offsets = svld1_u64(pg64, lane_offsets_buf);

    auto f = [](unsigned n, unsigned m_idx, uint64_t i_block, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                svuint64_t v_lane_offsets, fp_type* rstate) {
      constexpr unsigned gsize = 1 << N;
      uint64_t vl64 = svcntd();
      svbool_t pg64 = svptrue_b64();
      svbool_t pg32 = svptrue_b32();
      uint64_t vl = svcntw();

      i_block *= vl64;
      uint64_t ii_block = i_block & ms[0];
      uint64_t shifted = i_block;
      for (unsigned j = 1; j <= N; ++j) {
        shifted *= 2;
        ii_block |= shifted & ms[j];
      }

      fp_type* p0 = rstate + 2 * ii_block;

      alignas(64) fp_type tmp[4096];
      if constexpr (N > 2) {
        for (unsigned k = 0; k < gsize; ++k) {
          svfloat64_t gathered = svld1_gather_u64offset_f64(
              pg64, (const double*)(p0 + xss[k]), v_lane_offsets);
          svst1_f32(pg32, tmp + k * vl, svreinterpret_f32_f64(gathered));
        }
      }

      for (unsigned k = 0; k < gsize;) {
        if (gsize - k >= 4) {
          svfloat32_t rn0 = svdup_n_f32(0.0f);
          svfloat32_t rn1 = svdup_n_f32(0.0f);
          svfloat32_t rn2 = svdup_n_f32(0.0f);
          svfloat32_t rn3 = svdup_n_f32(0.0f);

          const double* p_v0 = (const double*)&v[2 * ((k + 0) * gsize)];
          const double* p_v1 = (const double*)&v[2 * ((k + 1) * gsize)];
          const double* p_v2 = (const double*)&v[2 * ((k + 2) * gsize)];
          const double* p_v3 = (const double*)&v[2 * ((k + 3) * gsize)];
#pragma GCC unroll 64
          for (unsigned l = 0; l < gsize; ++l) {
            svfloat32_t sv_l;
            if constexpr (N <= 2) {
              sv_l = svreinterpret_f32_f64(svld1_gather_u64offset_f64(
                  pg64, (const double*)(p0 + xss[l]), v_lane_offsets));
            } else {
              sv_l = svld1_f32(pg32, tmp + l * vl);
            }

            svfloat32_t m_kl0 = svreinterpret_f32_f64(svdup_n_f64(*p_v0++));
            svfloat32_t m_kl1 = svreinterpret_f32_f64(svdup_n_f64(*p_v1++));
            svfloat32_t m_kl2 = svreinterpret_f32_f64(svdup_n_f64(*p_v2++));
            svfloat32_t m_kl3 = svreinterpret_f32_f64(svdup_n_f64(*p_v3++));

            rn0 = svcmla_f32_x(pg32, rn0, sv_l, m_kl0, 0);
            rn0 = svcmla_f32_x(pg32, rn0, sv_l, m_kl0, 90);
            rn1 = svcmla_f32_x(pg32, rn1, sv_l, m_kl1, 0);
            rn1 = svcmla_f32_x(pg32, rn1, sv_l, m_kl1, 90);
            rn2 = svcmla_f32_x(pg32, rn2, sv_l, m_kl2, 0);
            rn2 = svcmla_f32_x(pg32, rn2, sv_l, m_kl2, 90);
            rn3 = svcmla_f32_x(pg32, rn3, sv_l, m_kl3, 0);
            rn3 = svcmla_f32_x(pg32, rn3, sv_l, m_kl3, 90);
          }

          QSIM_SVE_SCATTER_STEP(0)
          QSIM_SVE_SCATTER_STEP(1)
          QSIM_SVE_SCATTER_STEP(2)
          QSIM_SVE_SCATTER_STEP(3)
          k += 4;
        } else if (gsize - k >= 2) {
          svfloat32_t rn0 = svdup_n_f32(0.0f);
          svfloat32_t rn1 = svdup_n_f32(0.0f);

          const double* p_v0 = (const double*)&v[2 * ((k + 0) * gsize)];
          const double* p_v1 = (const double*)&v[2 * ((k + 1) * gsize)];
#pragma GCC unroll 64
          for (unsigned l = 0; l < gsize; ++l) {
            svfloat32_t sv_l;
            if constexpr (N <= 2) {
              sv_l = svreinterpret_f32_f64(svld1_gather_u64offset_f64(
                  pg64, (const double*)(p0 + xss[l]), v_lane_offsets));
            } else {
              sv_l = svld1_f32(pg32, tmp + l * vl);
            }

            svfloat32_t m_kl0 = svreinterpret_f32_f64(svdup_n_f64(*p_v0++));
            svfloat32_t m_kl1 = svreinterpret_f32_f64(svdup_n_f64(*p_v1++));

            rn0 = svcmla_f32_x(pg32, rn0, sv_l, m_kl0, 0);
            rn0 = svcmla_f32_x(pg32, rn0, sv_l, m_kl0, 90);
            rn1 = svcmla_f32_x(pg32, rn1, sv_l, m_kl1, 0);
            rn1 = svcmla_f32_x(pg32, rn1, sv_l, m_kl1, 90);
          }

          QSIM_SVE_SCATTER_STEP(0)
          QSIM_SVE_SCATTER_STEP(1)
          k += 2;
        } else {
          svfloat32_t rn0 = svdup_n_f32(0.0f);

          const double* p_v0 = (const double*)&v[2 * ((k + 0) * gsize)];
#pragma GCC unroll 64
          for (unsigned l = 0; l < gsize; ++l) {
            svfloat32_t sv_l;
            if constexpr (N <= 2) {
              sv_l = svreinterpret_f32_f64(svld1_gather_u64offset_f64(
                  pg64, (const double*)(p0 + xss[l]), v_lane_offsets));
            } else {
              sv_l = svld1_f32(pg32, tmp + l * vl);
            }

            QSIM_SVE_COMPUTE_STEP(0)
          }

          QSIM_SVE_SCATTER_STEP(0)
          k += 1;
        }
      }
    };

    const unsigned k_param = bits::Log2(vl64) + N;
    const unsigned n =
        state.num_qubits() > k_param ? state.num_qubits() - k_param : 0;
    const uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, v_lane_offsets, state.get());
  }

  template <unsigned N>
  void ApplyControlledGateCore(
      const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs,
      uint64_t cvals, const fp_type* matrix, State& state) const {
    constexpr unsigned gsize = 1 << N;
    uint64_t vl64 = svcntd();
    svbool_t pg64 = svptrue_b64();

    unsigned log_vl64 = bits::Log2(vl64);
    unsigned cl = 0;
    uint64_t cmaskh = 0;
    uint64_t cmaskl = 0;

    for (auto q : cqs) {
      if (q >= log_vl64) {
        cmaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        cmaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cvalsh = bits::ExpandBits(cvals >> cl, state.num_qubits(), cmaskh);
    uint64_t cvalsl =
        bits::ExpandBits(cvals & ((1 << cl) - 1), log_vl64, cmaskl);

    uint64_t ms[N + 1];
    uint64_t xss[gsize];
    FillIndices<N>(state.num_qubits(), qs, ms, xss);

    uint64_t lane_offsets_buf[64];
    for (uint64_t lane = 0; lane < vl64; ++lane) {
      uint64_t shifted = lane;
      uint64_t ii_lane = lane & ms[0];
      for (unsigned j = 1; j <= N; ++j) {
        shifted *= 2;
        ii_lane |= shifted & ms[j];
      }
      lane_offsets_buf[lane] = ii_lane * 2 * sizeof(fp_type);
    }
    svuint64_t v_lane_offsets = svld1_u64(pg64, lane_offsets_buf);

    auto f = [](unsigned n, unsigned m_idx, uint64_t i_block, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                svuint64_t v_lane_offsets, uint64_t cvalsh, uint64_t cmaskh,
                uint64_t cvalsl, uint64_t cmaskl, fp_type* rstate) {
      constexpr unsigned gsize = 1 << N;
      uint64_t vl64 = svcntd();
      svbool_t pg64 = svptrue_b64();
      svbool_t pg32 = svptrue_b32();
      uint64_t vl = svcntw();
      svuint32_t idx = svlsr_n_u32_x(pg32, svindex_u32(0, 1), 1);

      i_block *= vl64;
      uint64_t ii_block = i_block & ms[0];
      uint64_t shifted = i_block;
      for (unsigned j = 1; j <= N; ++j) {
        shifted *= 2;
        ii_block |= shifted & ms[j];
      }

      if ((ii_block & cmaskh) != cvalsh) return;

      fp_type* p0 = rstate + 2 * ii_block;

      alignas(64) fp_type tmp[4096];
      if constexpr (N > 2) {
        for (unsigned k = 0; k < gsize; ++k) {
          svfloat64_t gathered = svld1_gather_u64offset_f64(
              pg64, (const double*)(p0 + xss[k]), v_lane_offsets);
          svst1_f32(pg32, tmp + k * vl, svreinterpret_f32_f64(gathered));
        }
      }

      svuint32_t expanded = svdup_n_u32(0);
      for (unsigned b = 0; b < 16; ++b) {
        if ((cmaskl >> b) & 1) {
          uint32_t bit = bits::ExpandBits((uint64_t)b, 16, cmaskl);
          if ((cvalsl >> b) & 1) {
            expanded = svorr_u32_z(pg32, expanded, svdup_n_u32(bit));
          }
        }
      }
      svbool_t c_match = svcmpeq_u32(
          pg32, svand_u32_z(pg32, idx, svdup_n_u32(cmaskl)), expanded);

      for (unsigned k = 0; k < gsize;) {
        if (gsize - k >= 4) {
          svfloat32_t rn0 = svdup_n_f32(0.0f);
          svfloat32_t rn1 = svdup_n_f32(0.0f);
          svfloat32_t rn2 = svdup_n_f32(0.0f);
          svfloat32_t rn3 = svdup_n_f32(0.0f);

          const double* p_v0 = (const double*)&v[2 * ((k + 0) * gsize)];
          const double* p_v1 = (const double*)&v[2 * ((k + 1) * gsize)];
          const double* p_v2 = (const double*)&v[2 * ((k + 2) * gsize)];
          const double* p_v3 = (const double*)&v[2 * ((k + 3) * gsize)];
#pragma GCC unroll 64
          for (unsigned l = 0; l < gsize; ++l) {
            svfloat32_t sv_l;
            if constexpr (N <= 2) {
              sv_l = svreinterpret_f32_f64(svld1_gather_u64offset_f64(
                  pg64, (const double*)(p0 + xss[l]), v_lane_offsets));
            } else {
              sv_l = svld1_f32(pg32, tmp + l * vl);
            }

            svfloat32_t m_kl0 = svreinterpret_f32_f64(svdup_n_f64(*p_v0++));
            svfloat32_t m_kl1 = svreinterpret_f32_f64(svdup_n_f64(*p_v1++));
            svfloat32_t m_kl2 = svreinterpret_f32_f64(svdup_n_f64(*p_v2++));
            svfloat32_t m_kl3 = svreinterpret_f32_f64(svdup_n_f64(*p_v3++));

            rn0 = svcmla_f32_x(pg32, rn0, sv_l, m_kl0, 0);
            rn0 = svcmla_f32_x(pg32, rn0, sv_l, m_kl0, 90);
            rn1 = svcmla_f32_x(pg32, rn1, sv_l, m_kl1, 0);
            rn1 = svcmla_f32_x(pg32, rn1, sv_l, m_kl1, 90);
            rn2 = svcmla_f32_x(pg32, rn2, sv_l, m_kl2, 0);
            rn2 = svcmla_f32_x(pg32, rn2, sv_l, m_kl2, 90);
            rn3 = svcmla_f32_x(pg32, rn3, sv_l, m_kl3, 0);
            rn3 = svcmla_f32_x(pg32, rn3, sv_l, m_kl3, 90);
          }

          QSIM_SVE_CTRL_ORG_STEP(0)
          QSIM_SVE_CTRL_ORG_STEP(1)
          QSIM_SVE_CTRL_ORG_STEP(2)
          QSIM_SVE_CTRL_ORG_STEP(3)

          QSIM_SVE_SCATTER_STEP(0)
          QSIM_SVE_SCATTER_STEP(1)
          QSIM_SVE_SCATTER_STEP(2)
          QSIM_SVE_SCATTER_STEP(3)
          k += 4;
        } else if (gsize - k >= 2) {
          svfloat32_t rn0 = svdup_n_f32(0.0f);
          svfloat32_t rn1 = svdup_n_f32(0.0f);

          const double* p_v0 = (const double*)&v[2 * ((k + 0) * gsize)];
          const double* p_v1 = (const double*)&v[2 * ((k + 1) * gsize)];
#pragma GCC unroll 64
          for (unsigned l = 0; l < gsize; ++l) {
            svfloat32_t sv_l;
            if constexpr (N <= 2) {
              sv_l = svreinterpret_f32_f64(svld1_gather_u64offset_f64(
                  pg64, (const double*)(p0 + xss[l]), v_lane_offsets));
            } else {
              sv_l = svld1_f32(pg32, tmp + l * vl);
            }

            svfloat32_t m_kl0 = svreinterpret_f32_f64(svdup_n_f64(*p_v0++));
            svfloat32_t m_kl1 = svreinterpret_f32_f64(svdup_n_f64(*p_v1++));

            rn0 = svcmla_f32_x(pg32, rn0, sv_l, m_kl0, 0);
            rn0 = svcmla_f32_x(pg32, rn0, sv_l, m_kl0, 90);
            rn1 = svcmla_f32_x(pg32, rn1, sv_l, m_kl1, 0);
            rn1 = svcmla_f32_x(pg32, rn1, sv_l, m_kl1, 90);
          }

          QSIM_SVE_CTRL_ORG_STEP(0)
          QSIM_SVE_CTRL_ORG_STEP(1)

          QSIM_SVE_SCATTER_STEP(0)
          QSIM_SVE_SCATTER_STEP(1)
          k += 2;
        } else {
          svfloat32_t rn0 = svdup_n_f32(0.0f);

          const double* p_v0 = (const double*)&v[2 * ((k + 0) * gsize)];
#pragma GCC unroll 64
          for (unsigned l = 0; l < gsize; ++l) {
            svfloat32_t sv_l;
            if constexpr (N <= 2) {
              sv_l = svreinterpret_f32_f64(svld1_gather_u64offset_f64(
                  pg64, (const double*)(p0 + xss[l]), v_lane_offsets));
            } else {
              sv_l = svld1_f32(pg32, tmp + l * vl);
            }

            QSIM_SVE_COMPUTE_STEP(0)
          }

          svfloat32_t org0;
          if constexpr (N <= 2) {
            org0 = svreinterpret_f32_f64(svld1_gather_u64offset_f64(
                pg64, (const double*)(p0 + xss[k + 0]), v_lane_offsets));
          } else {
            org0 = svld1_f32(pg32, tmp + (k + 0) * vl);
          }
          rn0 = svsel_f32(c_match, rn0, org0);

          QSIM_SVE_SCATTER_STEP(0)
          k += 1;
        }
      }
    };

    const unsigned k_param = bits::Log2(vl64) + N + cqs.size();
    const unsigned n =
        state.num_qubits() > k_param ? state.num_qubits() - k_param : 0;
    const uint64_t size = uint64_t{1} << n;

    for_.Run(
        size, f, matrix, ms, xss, v_lane_offsets, cvalsh, cmaskh, cvalsl,
        cmaskl, state.get());
  }

  template <unsigned N>
  std::complex<double> ExpectationValueCore(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      const State& state) const {
    constexpr unsigned gsize = 1 << N;
    uint64_t vl64 = svcntd();
    svbool_t pg64 = svptrue_b64();

    uint64_t ms[N + 1];
    uint64_t xss[gsize];
    FillIndices<N>(state.num_qubits(), qs, ms, xss);

    uint64_t lane_offsets_buf[64];
    for (uint64_t lane = 0; lane < vl64; ++lane) {
      uint64_t shifted = lane;
      uint64_t ii_lane = lane & ms[0];
      for (unsigned j = 1; j <= N; ++j) {
        shifted *= 2;
        ii_lane |= shifted & ms[j];
      }
      lane_offsets_buf[lane] = ii_lane * 2 * sizeof(fp_type);
    }
    svuint64_t v_lane_offsets = svld1_u64(pg64, lane_offsets_buf);

    auto f = [](unsigned n, unsigned m_idx, uint64_t i_block, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                svuint64_t v_lane_offsets,
                const fp_type* rstate) -> std::complex<double> {
      constexpr unsigned gsize = 1 << N;
      uint64_t vl64 = svcntd();
      svbool_t pg64 = svptrue_b64();
      svbool_t pg32 = svptrue_b32();
      uint64_t vl = svcntw();

      i_block *= vl64;
      uint64_t ii_block = i_block & ms[0];
      uint64_t shifted = i_block;
      for (unsigned j = 1; j <= N; ++j) {
        shifted *= 2;
        ii_block |= shifted & ms[j];
      }

      const fp_type* p0 = rstate + 2 * ii_block;

      alignas(64) fp_type tmp[4096];
      if constexpr (N > 2) {
        for (unsigned k = 0; k < gsize; ++k) {
          svfloat64_t gathered = svld1_gather_u64offset_f64(
              pg64, (const double*)(p0 + xss[k]), v_lane_offsets);
          svst1_f32(pg32, tmp + k * vl, svreinterpret_f32_f64(gathered));
        }
      }

      svfloat32_t acc_re = svdup_n_f32(0.0f);
      svfloat32_t acc_im = svdup_n_f32(0.0f);

      for (unsigned k = 0; k < gsize; ++k) {
        svfloat32_t sum = svdup_n_f32(0.0f);
        const double* p_v0 = (const double*)&v[2 * (k * gsize)];
#pragma GCC unroll 64
        for (unsigned l = 0; l < gsize; ++l) {
          svfloat32_t m_kl = svreinterpret_f32_f64(svdup_n_f64(*p_v0++));

          svfloat32_t sv_l = svld1_f32(pg32, tmp + l * vl);
          sum = svcmla_f32_x(pg32, sum, sv_l, m_kl, 0);
          sum = svcmla_f32_x(pg32, sum, sv_l, m_kl, 90);
        }

        svfloat32_t rk;
        if constexpr (N <= 2) {
          rk = svreinterpret_f32_f64(svld1_gather_u64offset_f64(
              pg64, (const double*)(p0 + xss[k]), v_lane_offsets));
        } else {
          rk = svld1_f32(pg32, tmp + k * vl);
        }

        svfloat32_t rk_re = svuzp1_f32(rk, rk);
        svfloat32_t rk_im = svuzp2_f32(rk, rk);
        svfloat32_t sum_re = svuzp1_f32(sum, sum);
        svfloat32_t sum_im = svuzp2_f32(sum, sum);

        acc_re = svmla_f32_x(pg32, acc_re, rk_re, sum_re);
        acc_re = svmla_f32_x(pg32, acc_re, rk_im, sum_im);
        acc_im = svmla_f32_x(pg32, acc_im, rk_re, sum_im);
        acc_im = svmls_f32_x(pg32, acc_im, rk_im, sum_re);
      }

      return std::complex<double>(
          (double)svaddv_f32(pg32, acc_re) * 0.5,
          (double)svaddv_f32(pg32, acc_im) * 0.5);
    };

    const unsigned k_param = bits::Log2(vl64) + N;
    const unsigned n =
        state.num_qubits() > k_param ? state.num_qubits() - k_param : 0;
    const uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(
        size, f, Op(), matrix, ms, xss, v_lane_offsets, state.get());
  }

  For for_;
  BasicStateSpace basic_state_space_;
  SimulatorBasic<For, float> fallback_;
};

}  // namespace qsim

#endif  // SIMULATOR_SVE2_H_
