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

// Benchmark app for the gate-batched CUDA runner (prototype). Options match
// qsim_gate_batch.x where they apply; -l is the shared-memory tile size.

#include <unistd.h>

#include <complex>
#include <cstdlib>
#include <limits>
#include <string>

#include "../lib/circuit_qsim_parser.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/io_file.h"
#include "../lib/operation.h"
#include "../lib/run_qsim_gate_batch_cuda.h"

struct Options {
  std::string circuit_file;
  unsigned maxtime = std::numeric_limits<unsigned>::max();
  unsigned max_fused_size = 3;
  unsigned tile_qubits = 13;
  unsigned min_eviction_floor = 0;
  unsigned max_gate_seeds = 64;
  bool commute_diagonal_gates = false;
  unsigned lane_qubits = 0;
  unsigned verbosity = 0;
};

Options GetOptions(int argc, char* argv[]) {
  constexpr char usage[] = "usage:\n  ./qsim_gate_batch_cuda -c circuit "
                           "-d maxtime -f max_fused_size -l tile_qubits "
                           "-e min_eviction_floor -g max_gate_seeds "
                           "-x commute_diagonal_gates -n lane_qubits "
                           "-v verbosity\n";
  Options opt;
  int k;
  while ((k = getopt(argc, argv, "c:d:f:l:e:g:x:n:v:")) != -1) {
    switch (k) {
      case 'c': opt.circuit_file = optarg; break;
      case 'd': opt.maxtime = std::atoi(optarg); break;
      case 'f': opt.max_fused_size = std::atoi(optarg); break;
      case 'l': opt.tile_qubits = std::atoi(optarg); break;
      case 'e': opt.min_eviction_floor = std::atoi(optarg); break;
      case 'g': opt.max_gate_seeds = std::atoi(optarg); break;
      case 'x': opt.commute_diagonal_gates = std::atoi(optarg) != 0; break;
      case 'n': opt.lane_qubits = std::atoi(optarg); break;
      case 'v': opt.verbosity = std::atoi(optarg); break;
      default: qsim::IO::errorf(usage); exit(1);
    }
  }
  if (opt.circuit_file.empty()) {
    qsim::IO::errorf(usage);
    exit(1);
  }
  return opt;
}

int main(int argc, char* argv[]) {
  using namespace qsim;

  const auto opt = GetOptions(argc, argv);

  Circuit<Operation<float>> circuit;
  if (!CircuitQsimParser<IOFile>::FromFile(opt.maxtime, opt.circuit_file,
                                           circuit)) {
    return 1;
  }

  using Runner = QSimGateBatchRunnerCUDA<IO, MultiQubitGateFuser<IO>>;
  using StateSpace = Runner::StateSpace;

  StateSpace state_space(StateSpace::Parameter{});
  Runner::QubitMappedState state(state_space.Create(circuit.num_qubits));
  if (state_space.IsNull(state.state)) {
    IO::errorf("not enough memory: is the number of qubits too large?\n");
    return 1;
  }
  state_space.SetStateZero(state.state);

  Runner::Parameter param;
  param.max_fused_size = opt.max_fused_size;
  param.tile_qubits = opt.tile_qubits;
  param.min_eviction_floor = opt.min_eviction_floor;
  param.max_gate_seeds = opt.max_gate_seeds;
  param.commute_diagonal_gates = opt.commute_diagonal_gates;
  param.lane_qubits = opt.lane_qubits;
  param.verbosity = opt.verbosity;

  if (!Runner::Run(param, circuit, state)) return 1;

  const auto size = std::min(uint64_t{8}, uint64_t{1} << circuit.num_qubits);
  for (uint64_t i = 0; i < size; ++i) {
    const auto a = state.GetAmpl(state_space, i);
    IO::messagef("%llu:%16.8g%16.8g%16.8g\n", (unsigned long long) i,
                 std::real(a), std::imag(a), std::norm(a));
  }
  return 0;
}
