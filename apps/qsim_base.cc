// Copyright 2019 Google LLC. All Rights Reserved.
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

#include <unistd.h>

#include <algorithm>
#include <complex>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "../lib/circuit_qsim_parser.h"
#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/io_file.h"
#include "../lib/operation.h"
#include "../lib/run_qsim.h"
#include "../lib/simmux.h"
#include "../lib/util_cpu.h"

constexpr char usage[] = "usage:\n  ./qsim_base -c circuit -d maxtime "
                         "-s seed -t threads -f max_fused_size "
                         "-v verbosity -k tile_low_bits "
                         "-b tile_min_layer_ops -z\n";

struct Options {
  std::string circuit_file;
  unsigned maxtime = std::numeric_limits<unsigned>::max();
  unsigned seed = 1;
  unsigned num_threads = 1;
  unsigned max_fused_size = 2;
  unsigned verbosity = 0;
  unsigned tile_low_bits = 0;
  unsigned tile_min_layer_ops = 1;
  bool denormals_are_zeros = false;
};

Options GetOptions(int argc, char* argv[]) {
  Options opt;

  int k;

  while ((k = getopt(argc, argv, "c:d:s:t:f:v:k:b:z")) != -1) {
    switch (k) {
      case 'c':
        opt.circuit_file = optarg;
        break;
      case 'd':
        opt.maxtime = std::atoi(optarg);
        break;
      case 's':
        opt.seed = std::atoi(optarg);
        break;
      case 't':
        opt.num_threads = std::atoi(optarg);
        break;
      case 'f':
        opt.max_fused_size = std::atoi(optarg);
        break;
      case 'v':
        opt.verbosity = std::atoi(optarg);
        break;
      case 'k':
        opt.tile_low_bits = std::atoi(optarg);
        break;
      case 'b':
        opt.tile_min_layer_ops = std::atoi(optarg);
        break;
      case 'z':
        opt.denormals_are_zeros = true;
        break;
      default:
        qsim::IO::errorf(usage);
        exit(1);
    }
  }

  return opt;
}

bool ValidateOptions(const Options& opt) {
  if (opt.circuit_file.empty()) {
    qsim::IO::errorf("circuit file is not provided.\n");
    qsim::IO::errorf(usage);
    return false;
  }

  if (opt.tile_low_bits != 0 && opt.tile_low_bits < 8) {
    qsim::IO::errorf("tile_low_bits must be 0 or at least 8.\n");
    return false;
  }

  if (opt.tile_min_layer_ops == 0) {
    qsim::IO::errorf("tile_min_layer_ops must be at least 1.\n");
    return false;
  }

  return true;
}

template <typename StateSpace, typename State>
void PrintAmplitudes(
    unsigned num_qubits, const StateSpace& state_space, const State& state) {
  static constexpr char const* bits[8] = {
    "000", "001", "010", "011", "100", "101", "110", "111",
  };

  uint64_t size = std::min(uint64_t{8}, uint64_t{1} << num_qubits);
  unsigned s = 3 - std::min(unsigned{3}, num_qubits);

  for (uint64_t i = 0; i < size; ++i) {
    auto a = state_space.GetAmpl(state, i);
    qsim::IO::messagef("%s:%16.8g%16.8g%16.8g\n",
                       bits[i] + s, std::real(a), std::imag(a), std::norm(a));
  }
}

namespace {

struct LayeredTileStats {
  uint64_t fused_ops = 0;
  uint64_t tile_local_ops = 0;
  uint64_t full_state_ops = 0;
  uint64_t tile_layers = 0;
  uint64_t full_state_layers = 0;
  uint64_t skipped_tile_layers = 0;
  uint64_t skipped_tile_ops = 0;
  uint64_t tile_visits = 0;
  uint64_t max_tile_layer_ops = 0;
};

bool IsTileLocalQubits(const qsim::Qubits& qubits, unsigned tile_low_bits) {
  return std::all_of(
      qubits.begin(), qubits.end(),
      [tile_low_bits](unsigned q) { return q < tile_low_bits; });
}

template <typename Operation>
bool IsTileLocalOp(const Operation& op, unsigned tile_low_bits) {
  if (const auto* gate = qsim::OpGetAlternative<qsim::FusedGate<float>>(op)) {
    return !gate->matrix.empty() &&
           IsTileLocalQubits(gate->qubits, tile_low_bits);
  }

  if (const auto* gate = qsim::OpGetAlternative<qsim::Gate<float>>(op)) {
    return IsTileLocalQubits(gate->qubits, tile_low_bits);
  }

  if (const auto* gate =
          qsim::OpGetAlternative<qsim::ControlledGate<float>>(op)) {
    return IsTileLocalQubits(gate->qubits, tile_low_bits) &&
           IsTileLocalQubits(gate->controlled_by, tile_low_bits);
  }

  return false;
}

template <typename Simulator, typename Operation>
void ApplyTileLocalLayer(const std::vector<const Operation*>& ops,
                         unsigned num_threads, unsigned tile_low_bits,
                         typename Simulator::State& state) {
  using StateSpace = typename Simulator::StateSpace;

  const unsigned tile_qubits = std::min(tile_low_bits, state.num_qubits());
  const uint64_t num_tiles = uint64_t{1} << (state.num_qubits() - tile_qubits);
  const uint64_t tile_stride = StateSpace::MinSize(tile_qubits);

  qsim::For for_(num_threads);
  Simulator tile_simulator(1);

  auto apply_tile = [](unsigned n, unsigned m, uint64_t tile,
                       const std::vector<const Operation*>* ops,
                       const Simulator* simulator, float* state,
                       unsigned tile_qubits, uint64_t tile_stride) {
    auto tile_state =
        StateSpace::Create(state + tile * tile_stride, tile_qubits);
    for (const auto* op : *ops) {
      qsim::ApplyGate(*simulator, *op, tile_state);
    }
  };

  for_.Run(num_tiles, apply_tile, &ops, &tile_simulator, state.get(),
           tile_qubits, tile_stride);
}

template <typename Simulator, typename Fuser>
bool RunLayeredTileBatch(unsigned num_threads, unsigned tile_low_bits,
                         unsigned tile_min_layer_ops, unsigned seed,
                         unsigned max_fused_size, unsigned verbosity,
                         qsim::Circuit<qsim::Operation<float>> circuit,
                         const typename Simulator::StateSpace& state_space,
                         const Simulator& simulator,
                         typename Simulator::State& state) {
  double t0 = 0.0;
  double t1 = 0.0;

  if (verbosity > 1) {
    t0 = qsim::GetTime();
  }

  std::mt19937 rgen(seed);

  if (verbosity > 1) {
    t1 = qsim::GetTime();
    qsim::IO::messagef("init time is %g seconds.\n", t1 - t0);
    t0 = qsim::GetTime();
  }

  typename Fuser::Parameter param;
  param.max_fused_size = max_fused_size;

  auto fused_ops = Fuser::FuseGates(param, state.num_qubits(), circuit.ops);
  if (fused_ops.size() == 0 && circuit.ops.size() > 0) {
    return false;
  }

  LayeredTileStats stats;
  stats.fused_ops = fused_ops.size();

  if (verbosity > 1) {
    t1 = qsim::GetTime();
    qsim::IO::messagef("fuse time is %g seconds.\n", t1 - t0);
  }

  if (verbosity > 0) {
    t0 = qsim::GetTime();
  }

  const unsigned tile_qubits = std::min(tile_low_bits, state.num_qubits());
  const uint64_t tiles_per_layer =
      uint64_t{1} << (state.num_qubits() - tile_qubits);
  using FusedOp = typename decltype(fused_ops)::value_type;
  std::vector<const FusedOp*> tile_layer;
  bool in_full_state_layer = false;

  auto flush_tile_layer = [&]() -> bool {
    if (tile_layer.empty()) {
      return true;
    }

    stats.max_tile_layer_ops =
        std::max<uint64_t>(stats.max_tile_layer_ops, tile_layer.size());

    if (tile_layer.size() >= tile_min_layer_ops) {
      ++stats.tile_layers;
      stats.tile_visits += tiles_per_layer;
      ApplyTileLocalLayer<Simulator>(
          tile_layer, num_threads, tile_low_bits, state);
    } else {
      ++stats.skipped_tile_layers;
      stats.skipped_tile_ops += tile_layer.size();
      for (const auto* op : tile_layer) {
        if (!qsim::ApplyGate(state_space, simulator, *op, rgen, state)) {
          return false;
        }
      }
    }

    tile_layer.clear();
    return true;
  };

  for (const auto& op : fused_ops) {
    if (IsTileLocalOp(op, tile_low_bits)) {
      ++stats.tile_local_ops;
      tile_layer.push_back(&op);
      in_full_state_layer = false;
      continue;
    }

    if (!flush_tile_layer()) {
      return false;
    }

    if (!in_full_state_layer) {
      ++stats.full_state_layers;
      in_full_state_layer = true;
    }
    ++stats.full_state_ops;

    if (!qsim::ApplyGate(state_space, simulator, op, rgen, state)) {
      return false;
    }
  }

  if (!flush_tile_layer()) {
    return false;
  }

  if (verbosity > 0) {
    state_space.DeviceSync();
    double t2 = qsim::GetTime();
    qsim::IO::messagef("simu time is %g seconds.\n", t2 - t0);
  }

  if (verbosity > 1) {
    const uint64_t total_layers = stats.tile_layers +
                                  stats.skipped_tile_layers +
                                  stats.full_state_layers;

    qsim::IO::messagef(
        "layer stats: k=%u tile_qubits=%u tiles_per_layer=%llu "
        "min_layer_ops=%u layers=%llu fused_ops=%llu local_ops=%llu "
        "full_state_ops=%llu tile_layers=%llu full_state_layers=%llu "
        "skipped_tile_layers=%llu skipped_tile_ops=%llu tile_visits=%llu "
        "max_tile_layer_ops=%llu\n",
        tile_low_bits, tile_qubits,
        static_cast<unsigned long long>(tiles_per_layer),
        tile_min_layer_ops,
        static_cast<unsigned long long>(total_layers),
        static_cast<unsigned long long>(stats.fused_ops),
        static_cast<unsigned long long>(stats.tile_local_ops),
        static_cast<unsigned long long>(stats.full_state_ops),
        static_cast<unsigned long long>(stats.tile_layers),
        static_cast<unsigned long long>(stats.full_state_layers),
        static_cast<unsigned long long>(stats.skipped_tile_layers),
        static_cast<unsigned long long>(stats.skipped_tile_ops),
        static_cast<unsigned long long>(stats.tile_visits),
        static_cast<unsigned long long>(stats.max_tile_layer_ops));
  }

  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  using namespace qsim;

  auto opt = GetOptions(argc, argv);
  if (!ValidateOptions(opt)) {
    return 1;
  }

  Circuit<Operation<float>> circuit;
  if (!CircuitQsimParser<IOFile>::FromFile(opt.maxtime, opt.circuit_file,
                                           circuit)) {
    return 1;
  }

  if (opt.denormals_are_zeros) {
    SetFlushToZeroAndDenormalsAreZeros();
  }

  struct Factory {
    Factory(unsigned num_threads) : num_threads(num_threads) {}

    using Simulator = qsim::Simulator<For>;
    using StateSpace = Simulator::StateSpace;

    StateSpace CreateStateSpace() const {
      return StateSpace(num_threads);
    }

    Simulator CreateSimulator() const {
      return Simulator(num_threads);
    }

    unsigned num_threads;
  };

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Fuser = MultiQubitGateFuser<IO>;
  using Runner = QSimRunner<IO, Fuser, Factory>;

  StateSpace state_space = Factory(opt.num_threads).CreateStateSpace();
  State state = state_space.Create(circuit.num_qubits);

  if (state_space.IsNull(state)) {
    IO::errorf("not enough memory: is the number of qubits too large?\n");
    return 1;
  }

  state_space.SetStateZero(state);

  Runner::Parameter param;
  param.max_fused_size = opt.max_fused_size;
  param.seed = opt.seed;
  param.verbosity = opt.verbosity;

  const bool use_layered_tile =
      opt.tile_low_bits != 0 && opt.tile_low_bits < circuit.num_qubits;
  const bool success =
      use_layered_tile
          ? RunLayeredTileBatch<Simulator, Fuser>(
                opt.num_threads, opt.tile_low_bits, opt.tile_min_layer_ops,
                opt.seed, opt.max_fused_size, opt.verbosity, circuit,
                state_space, Factory(opt.num_threads).CreateSimulator(), state)
          : Runner::Run(param, Factory(opt.num_threads), circuit, state);

  if (success) {
    PrintAmplitudes(circuit.num_qubits, state_space, state);
  }

  return 0;
}
