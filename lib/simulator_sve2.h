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
#include <functional>
#include <vector>

#if !defined(__ARM_FEATURE_SVE2)
#error "simulator_sve2.h requires SVE2 support."
#endif
#include <arm_sve.h>

#include "simulator.h"
#include "statespace_sve.h"
#include "simulator_basic.h"
#include "statespace_basic.h"

namespace qsim {

/**
 * Quantum circuit simulator with SVE2 vectorization.
 */
template <typename For>
class SimulatorSVE2 final : public SimulatorBase {
 public:
  using StateSpace = StateSpaceSVE<For>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using BasicStateSpace = StateSpaceBasic<For, float>;
  using BasicState = typename BasicStateSpace::State;

  template <typename... ForArgs>
  explicit SimulatorSVE2(ForArgs&&... args) : for_(args...), basic_state_space_(args...), fallback_(args...) {}

  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, State& state) const {
    uint64_t vl = svcntw();
    unsigned nlb = bits::Log2(vl);

    switch (qs.size()) {
      case 1:
        if (qs[0] >= nlb) { ApplyGateH<1>(qs, matrix, state); }
        else { ApplyGateL<0, 1>(qs, matrix, state); }
        return;
      case 2:
        if (qs[0] >= nlb) { ApplyGateH<2>(qs, matrix, state); }
        else if (qs[1] >= nlb) { ApplyGateL<1, 1>(qs, matrix, state); }
        else { ApplyGateL<0, 2>(qs, matrix, state); }
        return;
      case 3:
        if (qs[0] >= nlb) { ApplyGateH<3>(qs, matrix, state); }
        else if (qs[1] >= nlb) { ApplyGateL<2, 1>(qs, matrix, state); }
        else if (qs[2] >= nlb) { ApplyGateL<1, 2>(qs, matrix, state); }
        else { ApplyGateL<0, 3>(qs, matrix, state); }
        return;
      case 4:
        if (qs[0] >= nlb) { ApplyGateH<4>(qs, matrix, state); }
        else if (qs[1] >= nlb) { ApplyGateL<3, 1>(qs, matrix, state); }
        else if (qs[2] >= nlb) { ApplyGateL<2, 2>(qs, matrix, state); }
        else if (qs[3] >= nlb) { ApplyGateL<1, 3>(qs, matrix, state); }
        else { ApplyGateL<0, 4>(qs, matrix, state); }
        return;
      case 5:
        if (qs[0] >= nlb) { ApplyGateH<5>(qs, matrix, state); }
        else if (qs[1] >= nlb) { ApplyGateL<4, 1>(qs, matrix, state); }
        else if (qs[2] >= nlb) { ApplyGateL<3, 2>(qs, matrix, state); }
        else if (qs[3] >= nlb) { ApplyGateL<2, 3>(qs, matrix, state); }
        else if (qs[4] >= nlb) { ApplyGateL<1, 4>(qs, matrix, state); }
        else { ApplyGateL<0, 5>(qs, matrix, state); }
        return;
      case 6:
        if (qs[0] >= nlb) { ApplyGateH<6>(qs, matrix, state); }
        else if (qs[1] >= nlb) { ApplyGateL<5, 1>(qs, matrix, state); }
        else if (qs[2] >= nlb) { ApplyGateL<4, 2>(qs, matrix, state); }
        else if (qs[3] >= nlb) { ApplyGateL<3, 3>(qs, matrix, state); }
        else if (qs[4] >= nlb) { ApplyGateL<2, 4>(qs, matrix, state); }
        else if (qs[5] >= nlb) { ApplyGateL<1, 5>(qs, matrix, state); }
        else { ApplyGateL<0, 6>(qs, matrix, state); }
        return;
      default: break;
    }
  }

  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cvals,
                           const fp_type* matrix, State& state) const {
    if (cqs.empty()) {
      ApplyGate(qs, matrix, state);
      return;
    }
    ApplyControlledGateFallback(qs, cqs, cvals, matrix, state);
  }

  std::complex<double> ExpectationValue(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      const State& state) const {
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

  template <unsigned H>
  void ApplyGateH(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;
      uint64_t vl = svcntw();
      svbool_t pg = svptrue_b32();

      i *= vl;
      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2; ii |= i & ms[j];
      }
      auto p0 = rstate + 2 * ii;
      __builtin_prefetch(p0 + 64);

      alignas(64) fp_type tmp_rs[4096];
      alignas(64) fp_type tmp_is[4096];

      for (unsigned k = 0; k < hsize; ++k) {
        svst1_f32(pg, tmp_rs + k * vl, svld1_f32(pg, p0 + xss[k]));
        svst1_f32(pg, tmp_is + k * vl, svld1_f32(pg, p0 + xss[k] + vl));
      }

      for (unsigned k = 0; k < hsize; ++k) {
        svfloat32_t rn = svdup_n_f32(0);
        svfloat32_t in = svdup_n_f32(0);

        for (unsigned l = 0; l < hsize; ++l) {
          svfloat32_t rl = svld1_f32(pg, tmp_rs + l * vl);
          svfloat32_t il = svld1_f32(pg, tmp_is + l * vl);

          svfloat32_t ru = svdup_n_f32(v[2 * (k * hsize + l)]);
          svfloat32_t iu = svdup_n_f32(v[2 * (k * hsize + l) + 1]);

          rn = svmla_f32_x(pg, rn, rl, ru);
          rn = svmls_f32_x(pg, rn, il, iu);
          in = svmla_f32_x(pg, in, rl, iu);
          in = svmla_f32_x(pg, in, il, ru);
        }

        svst1_f32(pg, p0 + xss[k], rn);
        svst1_f32(pg, p0 + xss[k] + vl, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    uint64_t vl = svcntw();
    const unsigned k_param = bits::Log2(vl) + H;
    const unsigned n = state.num_qubits() > k_param ? state.num_qubits() - k_param : 0;
    const uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, state.get());
  }

  template <unsigned H, unsigned L>
  void ApplyGateL(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      State& state) const {
    constexpr unsigned gsize = 1 << (H + L);
    constexpr unsigned hsize = 1 << H;
    constexpr unsigned lsize = 1 << L;

    uint64_t vl = svcntw();
    unsigned nlb = bits::Log2(vl);
    unsigned q0 = qs[0];

    auto m = GetMasks11<L>(qs);
    uint64_t qmaskl = m.qmaskl;

    std::vector<fp_type> precomp_wre(hsize * gsize * vl);
    std::vector<fp_type> precomp_wim(hsize * gsize * vl);

    for (unsigned k = 0; k < hsize; ++k) {
      for (unsigned l = 0; l < gsize; ++l) {
        unsigned h_idx = l / lsize;
        unsigned r_idx = l % lsize;
        for (unsigned lane = 0; lane < vl; ++lane) {
          unsigned s_out = bits::CompressBits((uint64_t)lane, nlb, qmaskl);
          unsigned row = lsize * k + s_out;
          unsigned col = lsize * h_idx + (s_out ^ r_idx);
          precomp_wre[(k * gsize + l) * vl + lane] = matrix[2 * (row * gsize + col)];
          precomp_wim[(k * gsize + l) * vl + lane] = matrix[2 * (row * gsize + col) + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m_idx, uint64_t i, const fp_type* wre_ptr,
                const fp_type* wim_ptr, const uint64_t* ms, const uint64_t* xss,
                uint64_t qmaskl, fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      svbool_t pg = svptrue_b32();
      uint64_t vl = svcntw();
      svuint32_t idx = svindex_u32(0, 1);
      unsigned nlb = bits::Log2(vl);

      i *= vl;
      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2; ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;
      __builtin_prefetch(p0 + 64);

      alignas(64) fp_type tmp_rs[4096];
      alignas(64) fp_type tmp_is[4096];

      uint32_t flips[64];
      for (unsigned r = 0; r < lsize; ++r) {
        flips[r] = bits::ExpandBits((uint64_t)r, nlb, qmaskl);
      }

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;
        svfloat32_t r0 = svld1_f32(pg, p0 + xss[k]);
        svfloat32_t i0 = svld1_f32(pg, p0 + xss[k] + vl);

        for (unsigned r = 0; r < lsize; ++r) {
          svuint32_t perm = sveor_u32_z(pg, idx, svdup_n_u32(flips[r]));
          svst1_f32(pg, tmp_rs + (k2 + r) * vl, svtbl_f32(r0, perm));
          svst1_f32(pg, tmp_is + (k2 + r) * vl, svtbl_f32(i0, perm));
        }
      }

      for (unsigned k = 0; k < hsize; ++k) {
        svfloat32_t rn = svdup_n_f32(0);
        svfloat32_t in = svdup_n_f32(0);

        for (unsigned l = 0; l < gsize; ++l) {
          svfloat32_t rl = svld1_f32(pg, tmp_rs + l * vl);
          svfloat32_t il = svld1_f32(pg, tmp_is + l * vl);

          svfloat32_t wre = svld1_f32(pg, wre_ptr + (k * gsize + l) * vl);
          svfloat32_t wim = svld1_f32(pg, wim_ptr + (k * gsize + l) * vl);

          rn = svmla_f32_x(pg, rn, rl, wre);
          rn = svmls_f32_x(pg, rn, il, wim);
          in = svmla_f32_x(pg, in, rl, wim);
          in = svmla_f32_x(pg, in, il, wre);
        }

        svst1_f32(pg, p0 + xss[k], rn);
        svst1_f32(pg, p0 + xss[k] + vl, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);

    const unsigned k_param = bits::Log2(vl) + H;
    const unsigned n = state.num_qubits() > k_param ? state.num_qubits() - k_param : 0;
    const uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, precomp_wre.data(), precomp_wim.data(), ms, xss, qmaskl, state.get());
  }

  template <unsigned H>
  void ApplyControlledGateH(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      uint64_t cvalsh, uint64_t cmaskh, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
                uint64_t cmaskh, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;
      uint64_t vl = svcntw();
      svbool_t pg = svptrue_b32();

      i *= vl;
      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2; ii |= i & ms[j];
      }

      if ((ii & cmaskh) != cvalsh) return;

      auto p0 = rstate + 2 * ii;
      __builtin_prefetch(p0 + 64);

      alignas(64) fp_type tmp_rs[4096];
      alignas(64) fp_type tmp_is[4096];

      for (unsigned k = 0; k < hsize; ++k) {
        svst1_f32(pg, tmp_rs + k * vl, svld1_f32(pg, p0 + xss[k]));
        svst1_f32(pg, tmp_is + k * vl, svld1_f32(pg, p0 + xss[k] + vl));
      }

      for (unsigned k = 0; k < hsize; ++k) {
        svfloat32_t rn = svdup_n_f32(0);
        svfloat32_t in = svdup_n_f32(0);

        for (unsigned l = 0; l < hsize; ++l) {
          svfloat32_t rl = svld1_f32(pg, tmp_rs + l * vl);
          svfloat32_t il = svld1_f32(pg, tmp_is + l * vl);

          svfloat32_t ru = svdup_n_f32(v[2 * (k * hsize + l)]);
          svfloat32_t iu = svdup_n_f32(v[2 * (k * hsize + l) + 1]);

          rn = svmla_f32_x(pg, rn, rl, ru);
          rn = svmls_f32_x(pg, rn, il, iu);
          in = svmla_f32_x(pg, in, rl, iu);
          in = svmla_f32_x(pg, in, il, ru);
        }

        svst1_f32(pg, p0 + xss[k], rn);
        svst1_f32(pg, p0 + xss[k] + vl, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    uint64_t vl = svcntw();
    const unsigned k_param = bits::Log2(vl) + H;
    const unsigned n = state.num_qubits() > k_param ? state.num_qubits() - k_param : 0;
    const uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, cvalsh, cmaskh, state.get());
  }

  template <unsigned H>
  void ApplyControlledGateHL(
      const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs,
      uint64_t cvals, const fp_type* matrix, uint64_t cvalsh, uint64_t cmaskh,
      State& state) const {
    auto f = [](unsigned n, unsigned m_idx, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
                uint64_t cmaskh, uint64_t cvalsl, uint64_t cmaskl,
                fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;
      uint64_t vl = svcntw();
      svbool_t pg = svptrue_b32();
      svuint32_t idx = svindex_u32(0, 1);
      unsigned nlb = bits::Log2(vl);

      i *= vl;
      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2; ii |= i & ms[j];
      }

      if ((ii & cmaskh) != cvalsh) return;

      auto p0 = rstate + 2 * ii;
      __builtin_prefetch(p0 + 64);

      alignas(64) fp_type tmp_rs[4096];
      alignas(64) fp_type tmp_is[4096];

      for (unsigned k = 0; k < hsize; ++k) {
        svst1_f32(pg, tmp_rs + k * vl, svld1_f32(pg, p0 + xss[k]));
        svst1_f32(pg, tmp_is + k * vl, svld1_f32(pg, p0 + xss[k] + vl));
      }

      svuint32_t expanded = svdup_n_u32(0);
      for (unsigned b = 0; b < 16; ++b) {
        if ((cmaskl >> b) & 1) {
          uint32_t bit = bits::ExpandBits((uint64_t)b, 16, cmaskl);
          if ((cvalsl >> b) & 1) {
            expanded = svorr_u32_z(pg, expanded, svdup_n_u32(bit));
          }
        }
      }
      svbool_t c_match = svcmpeq_u32(pg, svand_u32_z(pg, idx, svdup_n_u32(cmaskl)), expanded);

      for (unsigned k = 0; k < hsize; ++k) {
        svfloat32_t rn = svld1_f32(pg, tmp_rs + k * vl);
        svfloat32_t in = svld1_f32(pg, tmp_is + k * vl);

        svfloat32_t rn_new = svdup_n_f32(0);
        svfloat32_t in_new = svdup_n_f32(0);

        for (unsigned l = 0; l < hsize; ++l) {
          svfloat32_t rl = svld1_f32(pg, tmp_rs + l * vl);
          svfloat32_t il = svld1_f32(pg, tmp_is + l * vl);

          svfloat32_t ru = svdup_n_f32(v[2 * (k * hsize + l)]);
          svfloat32_t iu = svdup_n_f32(v[2 * (k * hsize + l) + 1]);

          rn_new = svmla_f32_x(pg, rn_new, rl, ru);
          rn_new = svmls_f32_x(pg, rn_new, il, iu);
          in_new = svmla_f32_x(pg, in_new, rl, iu);
          in_new = svmla_f32_x(pg, in_new, il, ru);
        }

        rn = svsel_f32(c_match, rn_new, rn);
        in = svsel_f32(c_match, in_new, in);

        svst1_f32(pg, p0 + xss[k], rn);
        svst1_f32(pg, p0 + xss[k] + vl, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    auto m = GetMasks8(state.num_qubits(), qs, cqs, cvals);

    uint64_t vl = svcntw();
    const unsigned k_param = bits::Log2(vl) + H;
    const unsigned n = state.num_qubits() > k_param ? state.num_qubits() - k_param : 0;
    const uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, m.cvalsh, m.cmaskh, m.cvalsl, m.cmaskl, state.get());
  }

  template <unsigned H, unsigned L, bool CH>
  void ApplyControlledGateL(
      const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs,
      uint64_t cvals, const fp_type* matrix, State& state) const {
    constexpr unsigned gsize = 1 << (H + L);
    constexpr unsigned hsize = 1 << H;
    constexpr unsigned lsize = 1 << L;

    uint64_t vl = svcntw();
    unsigned nlb = bits::Log2(vl);
    unsigned q0 = qs[0];

    auto m = GetMasks9<L>(state.num_qubits(), qs, cqs, cvals);
    uint64_t qmaskl = m.qmaskl;

    std::vector<fp_type> precomp_wre(hsize * gsize * vl);
    std::vector<fp_type> precomp_wim(hsize * gsize * vl);

    for (unsigned k = 0; k < hsize; ++k) {
      for (unsigned l = 0; l < gsize; ++l) {
        unsigned h_idx = l / lsize;
        unsigned r_idx = l % lsize;
        for (unsigned lane = 0; lane < vl; ++lane) {
          unsigned s_out = bits::CompressBits((uint64_t)lane, nlb, qmaskl);
          unsigned row = lsize * k + s_out;
          unsigned col = lsize * h_idx + (s_out ^ r_idx);
          precomp_wre[(k * gsize + l) * vl + lane] = matrix[2 * (row * gsize + col)];
          precomp_wim[(k * gsize + l) * vl + lane] = matrix[2 * (row * gsize + col) + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m_idx, uint64_t i, const fp_type* wre_ptr,
                const fp_type* wim_ptr, const uint64_t* ms, const uint64_t* xss,
                uint64_t qmaskl, uint64_t cvalsh, uint64_t cmaskh, uint64_t cvalsl, uint64_t cmaskl,
                fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      svbool_t pg = svptrue_b32();
      uint64_t vl = svcntw();
      svuint32_t idx = svindex_u32(0, 1);
      unsigned nlb = bits::Log2(vl);

      i *= vl;
      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2; ii |= i & ms[j];
      }

      if ((ii & cmaskh) != cvalsh) return;

      auto p0 = rstate + 2 * ii;
      __builtin_prefetch(p0 + 64);

      alignas(64) fp_type tmp_rs[4096];
      alignas(64) fp_type tmp_is[4096];

      uint32_t flips[64];
      for (unsigned r = 0; r < lsize; ++r) {
        flips[r] = bits::ExpandBits((uint64_t)r, nlb, qmaskl);
      }

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;
        svfloat32_t r0 = svld1_f32(pg, p0 + xss[k]);
        svfloat32_t i0 = svld1_f32(pg, p0 + xss[k] + vl);

        for (unsigned r = 0; r < lsize; ++r) {
          svuint32_t perm = sveor_u32_z(pg, idx, svdup_n_u32(flips[r]));
          svst1_f32(pg, tmp_rs + (k2 + r) * vl, svtbl_f32(r0, perm));
          svst1_f32(pg, tmp_is + (k2 + r) * vl, svtbl_f32(i0, perm));
        }
      }

      svuint32_t expanded = svdup_n_u32(0);
      for (unsigned b = 0; b < 16; ++b) {
        if ((cmaskl >> b) & 1) {
          uint32_t bit = bits::ExpandBits((uint64_t)b, 16, cmaskl);
          if ((cvalsl >> b) & 1) {
            expanded = svorr_u32_z(pg, expanded, svdup_n_u32(bit));
          }
        }
      }
      svbool_t c_match = svcmpeq_u32(pg, svand_u32_z(pg, idx, svdup_n_u32(cmaskl)), expanded);

      for (unsigned k = 0; k < hsize; ++k) {
        svfloat32_t rn = svld1_f32(pg, tmp_rs + k * lsize * vl);
        svfloat32_t in = svld1_f32(pg, tmp_is + k * lsize * vl);

        svfloat32_t rn_new = svdup_n_f32(0);
        svfloat32_t in_new = svdup_n_f32(0);

        for (unsigned l = 0; l < gsize; ++l) {
          svfloat32_t rl = svld1_f32(pg, tmp_rs + l * vl);
          svfloat32_t il = svld1_f32(pg, tmp_is + l * vl);

          svfloat32_t wre = svld1_f32(pg, wre_ptr + (k * gsize + l) * vl);
          svfloat32_t wim = svld1_f32(pg, wim_ptr + (k * gsize + l) * vl);

          rn_new = svmla_f32_x(pg, rn_new, rl, wre);
          rn_new = svmls_f32_x(pg, rn_new, il, wim);
          in_new = svmla_f32_x(pg, in_new, rl, wim);
          in_new = svmla_f32_x(pg, in_new, il, wre);
        }

        rn = svsel_f32(c_match, rn_new, rn);
        in = svsel_f32(c_match, in_new, in);

        svst1_f32(pg, p0 + xss[k], rn);
        svst1_f32(pg, p0 + xss[k] + vl, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);

    uint64_t cvalsl = m.cvalsl;
    uint64_t cmaskl = m.cmaskl;

    const unsigned k_param = bits::Log2(vl) + H;
    const unsigned n = state.num_qubits() > k_param ? state.num_qubits() - k_param : 0;
    const uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, precomp_wre.data(), precomp_wim.data(), ms, xss, qmaskl, m.cvalsh, m.cmaskh, cvalsl, cmaskl, state.get());
  }

  template <unsigned H>
  std::complex<double> ExpectationValueH(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) -> std::complex<double> {
      constexpr unsigned hsize = 1 << H;
      uint64_t vl = svcntw();
      svbool_t pg = svptrue_b32();

      i *= vl;
      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2; ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;
      __builtin_prefetch(p0 + 64);

      alignas(64) fp_type tmp_rs[4096];
      alignas(64) fp_type tmp_is[4096];

      for (unsigned k = 0; k < hsize; ++k) {
        svst1_f32(pg, tmp_rs + k * vl, svld1_f32(pg, p0 + xss[k]));
        svst1_f32(pg, tmp_is + k * vl, svld1_f32(pg, p0 + xss[k] + vl));
      }

      svfloat32_t acc_re = svdup_n_f32(0);
      svfloat32_t acc_im = svdup_n_f32(0);

      for (unsigned k = 0; k < hsize; ++k) {
        svfloat32_t rn = svdup_n_f32(0);
        svfloat32_t in = svdup_n_f32(0);

        for (unsigned l = 0; l < hsize; ++l) {
          svfloat32_t rl = svld1_f32(pg, tmp_rs + l * vl);
          svfloat32_t il = svld1_f32(pg, tmp_is + l * vl);

          svfloat32_t ru = svdup_n_f32(v[2 * (k * hsize + l)]);
          svfloat32_t iu = svdup_n_f32(v[2 * (k * hsize + l) + 1]);

          rn = svmla_f32_x(pg, rn, rl, ru);
          rn = svmls_f32_x(pg, rn, il, iu);
          in = svmla_f32_x(pg, in, rl, iu);
          in = svmla_f32_x(pg, in, il, ru);
        }

        svfloat32_t rk = svld1_f32(pg, tmp_rs + k * vl);
        svfloat32_t ik = svld1_f32(pg, tmp_is + k * vl);

        acc_re = svmla_f32_x(pg, acc_re, rk, rn);
        acc_re = svmla_f32_x(pg, acc_re, ik, in);
        acc_im = svmla_f32_x(pg, acc_im, rk, in);
        acc_im = svmls_f32_x(pg, acc_im, ik, rn);
      }

      return std::complex<double>((double)svaddv_f32(pg, acc_re),
                                  (double)svaddv_f32(pg, acc_im));
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    uint64_t vl = svcntw();
    const unsigned k_param = bits::Log2(vl) + H;
    const unsigned n = state.num_qubits() > k_param ? state.num_qubits() - k_param : 0;
    const uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, state.get());
  }

  template <unsigned H, unsigned L>
  std::complex<double> ExpectationValueL(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      const State& state) const {
    constexpr unsigned gsize = 1 << (H + L);
    constexpr unsigned hsize = 1 << H;
    constexpr unsigned lsize = 1 << L;

    uint64_t vl = svcntw();
    unsigned nlb = bits::Log2(vl);
    unsigned q0 = qs[0];

    auto m = GetMasks11<L>(qs);
    uint64_t qmaskl = m.qmaskl;

    std::vector<fp_type> precomp_wre(hsize * gsize * vl);
    std::vector<fp_type> precomp_wim(hsize * gsize * vl);

    for (unsigned k = 0; k < hsize; ++k) {
      for (unsigned l = 0; l < gsize; ++l) {
        unsigned h_idx = l / lsize;
        unsigned r_idx = l % lsize;
        for (unsigned lane = 0; lane < vl; ++lane) {
          unsigned s_out = bits::CompressBits((uint64_t)lane, nlb, qmaskl);
          unsigned row = lsize * k + s_out;
          unsigned col = lsize * h_idx + (s_out ^ r_idx);
          precomp_wre[(k * gsize + l) * vl + lane] = matrix[2 * (row * gsize + col)];
          precomp_wim[(k * gsize + l) * vl + lane] = matrix[2 * (row * gsize + col) + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m_idx, uint64_t i, const fp_type* wre_ptr,
                const fp_type* wim_ptr, const uint64_t* ms, const uint64_t* xss,
                uint64_t qmaskl, const fp_type* rstate) -> std::complex<double> {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      svbool_t pg = svptrue_b32();
      uint64_t vl = svcntw();
      svuint32_t idx = svindex_u32(0, 1);
      unsigned nlb = bits::Log2(vl);

      i *= vl;
      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2; ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;
      __builtin_prefetch(p0 + 64);

      alignas(64) fp_type tmp_rs[4096];
      alignas(64) fp_type tmp_is[4096];

      uint32_t flips[64];
      for (unsigned r = 0; r < lsize; ++r) {
        flips[r] = bits::ExpandBits((uint64_t)r, nlb, qmaskl);
      }

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;
        svfloat32_t r0 = svld1_f32(pg, p0 + xss[k]);
        svfloat32_t i0 = svld1_f32(pg, p0 + xss[k] + vl);

        for (unsigned r = 0; r < lsize; ++r) {
          svuint32_t perm = sveor_u32_z(pg, idx, svdup_n_u32(flips[r]));
          svst1_f32(pg, tmp_rs + (k2 + r) * vl, svtbl_f32(r0, perm));
          svst1_f32(pg, tmp_is + (k2 + r) * vl, svtbl_f32(i0, perm));
        }
      }

      svfloat32_t acc_re = svdup_n_f32(0);
      svfloat32_t acc_im = svdup_n_f32(0);

      for (unsigned k = 0; k < hsize; ++k) {
        svfloat32_t rn = svdup_n_f32(0);
        svfloat32_t in = svdup_n_f32(0);

        for (unsigned l = 0; l < gsize; ++l) {
          svfloat32_t rl = svld1_f32(pg, tmp_rs + l * vl);
          svfloat32_t il = svld1_f32(pg, tmp_is + l * vl);

          svfloat32_t wre = svld1_f32(pg, wre_ptr + (k * gsize + l) * vl);
          svfloat32_t wim = svld1_f32(pg, wim_ptr + (k * gsize + l) * vl);

          rn = svmla_f32_x(pg, rn, rl, wre);
          rn = svmls_f32_x(pg, rn, il, wim);
          in = svmla_f32_x(pg, in, rl, wim);
          in = svmla_f32_x(pg, in, il, wre);
        }

        svfloat32_t rk = svld1_f32(pg, tmp_rs + k * lsize * vl);
        svfloat32_t ik = svld1_f32(pg, tmp_is + k * lsize * vl);

        acc_re = svmla_f32_x(pg, acc_re, rk, rn);
        acc_re = svmla_f32_x(pg, acc_re, ik, in);
        acc_im = svmla_f32_x(pg, acc_im, rk, in);
        acc_im = svmls_f32_x(pg, acc_im, ik, rn);
      }

      return std::complex<double>((double)svaddv_f32(pg, acc_re),
                                  (double)svaddv_f32(pg, acc_im));
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);

    const unsigned k_param = bits::Log2(vl) + H;
    const unsigned n = state.num_qubits() > k_param ? state.num_qubits() - k_param : 0;
    const uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), precomp_wre.data(), precomp_wim.data(), ms, xss, qmaskl, state.get());
  }

  For for_;
  BasicStateSpace basic_state_space_;
  SimulatorBasic<For, float> fallback_;
};

}  // namespace qsim

#endif  // SIMULATOR_SVE2_H_
