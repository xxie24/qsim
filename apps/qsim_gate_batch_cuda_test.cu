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

// Correctness test: qsim's standard CUDA runner vs the gate-batched CUDA
// runner on the same circuit; compares the full final states.
//
// usage: ./qsim_gate_batch_cuda_test.x circuit_file tile_qubits [fused]
//                                      [lane_qubits] [min_eviction_floor]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "../lib/circuit_qsim_parser.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/io_file.h"
#include "../lib/operation.h"
#include "../lib/run_qsim.h"
#include "../lib/run_qsim_gate_batch_cuda.h"
#include "../lib/simulator_cuda.h"

int main(int argc, char* argv[]) {
  using namespace qsim;

  if (argc < 3) {
    IO::errorf("usage: %s circuit_file tile_qubits [max_fused_size]\n",
               argv[0]);
    return 1;
  }
  const unsigned tile_qubits = std::atoi(argv[2]);
  const unsigned max_fused_size = argc > 3 ? std::atoi(argv[3]) : 4;
  const unsigned lane_qubits = argc > 4 ? std::atoi(argv[4]) : 5;
  const unsigned min_eviction_floor = argc > 5 ? std::atoi(argv[5]) : 5;

  Circuit<Operation<float>> circuit;
  if (!CircuitQsimParser<IOFile>::FromFile(
          std::numeric_limits<unsigned>::max(), argv[1], circuit)) {
    return 1;
  }

  struct Factory {
    using Simulator = SimulatorCUDA<float>;
    using StateSpace = Simulator::StateSpace;
    StateSpace CreateStateSpace() const { return StateSpace(param); }
    Simulator CreateSimulator() const { return Simulator(); }
    StateSpace::Parameter param;
  };
  using Fuser = MultiQubitGateFuser<IO>;
  using LegacyRunner = QSimRunner<IO, Fuser, Factory>;
  using BatchRunner = QSimGateBatchRunnerCUDA<IO, Fuser>;

  Factory factory;
  auto state_space = factory.CreateStateSpace();
  auto state1 = state_space.Create(circuit.num_qubits);
  BatchRunner::QubitMappedState state2(state_space.Create(circuit.num_qubits));
  if (state_space.IsNull(state1) || state_space.IsNull(state2.state)) {
    IO::errorf("not enough GPU memory\n");
    return 1;
  }
  state_space.SetStateZero(state1);
  state_space.SetStateZero(state2.state);

  LegacyRunner::Parameter lp;
  lp.max_fused_size = max_fused_size;
  if (!LegacyRunner::Run(lp, factory, circuit, state1)) return 1;

  BatchRunner::Parameter bp;
  bp.max_fused_size = max_fused_size;
  bp.tile_qubits = tile_qubits;
  bp.lane_qubits = lane_qubits;
  bp.min_eviction_floor = min_eviction_floor;
  if (!BatchRunner::Run(bp, circuit, state2)) return 1;

  const auto num_floats = Factory::StateSpace::MinSize(circuit.num_qubits);
  std::vector<float> h1(num_floats), h2(num_floats);
  cudaMemcpy(h1.data(), state1.get(), num_floats * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h2.data(), state2.state.get(), num_floats * 4,
             cudaMemcpyDeviceToHost);

  double max_diff = 0;
  uint64_t bad = 0;
  const uint64_t num_amplitudes = uint64_t{1} << circuit.num_qubits;
  using remap_cuda::kLanes;
  using remap_cuda::RealFloatIndex;
  for (uint64_t i = 0; i < num_amplitudes; ++i) {
    const auto f1 = RealFloatIndex(i);
    const auto f2 = RealFloatIndex(state2.layout.PhysicalAmplitudeIndex(i));
    const double d =
        std::max(std::fabs(double(h1[f1]) - h2[f2]),
                 std::fabs(double(h1[f1 + kLanes]) - h2[f2 + kLanes]));
    max_diff = std::max(max_diff, d);
    if (d > 1e-5) ++bad;
  }

  const bool pass = max_diff < 1e-4;
  std::printf("gate-batch-cuda: n=%u L=%u f=%u lanes=%u e=%u  max_diff=%.3e  "
              "bad=%llu  %s\n", circuit.num_qubits, tile_qubits,
              max_fused_size, lane_qubits, min_eviction_floor, max_diff,
              (unsigned long long) bad, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
