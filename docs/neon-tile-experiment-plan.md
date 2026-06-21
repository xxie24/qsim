# NEON Tile Execution Experiment Plan

This note captures a concrete experiment plan for exploring tile-based
execution in qsim using the NEON backend on the current `main` branch.

## Goal

The goal is to understand whether tile-based execution can improve qsim
state-vector performance on NEON by increasing cache reuse, especially for
blocks of fused gates that act only on low physical qubits.

This plan intentionally separates:

1. baseline fused execution
2. tiled execution without remapping
3. tiled execution with remapping

so we can see clearly which idea is responsible for any improvement.

## 1. Establish a Clean NEON Baseline

Record the exact baseline setup on current `main`:

- build command
- run command
- backend used
- target circuits
- thread counts
- fusion settings
- correctness output to compare against later

Suggested initial targets:

- `circuit_q24`
- `circuit_q30`

Suggested initial knobs to record:

- `-t` thread count
- `-f` fused gate size
- `-v` verbosity

The purpose is to make every future measurement answer:

```text
better than what baseline?
```

### Baseline Checkpoint: 2026-06-21

Repository state:

```text
/Users/xxie/src/qsim
branch: main
status: clean relative to upstream/main
binary: apps/qsim_base.x, Mach-O arm64
backend: NEON through lib/simmux.h on arm64
```

The current `simmux.h` selects `SimulatorNEON<For>` when `__ARM_NEON__` or
`__ARM_NEON` is defined, so the arm64 binary is expected to use the NEON
backend.

Baseline proxy runs:

```text
./apps/qsim_base.x -c ./circuits/circuit_q24 -t 1 -f 3 -v 1 -d 12
simu time is 0.46834 seconds.

./apps/qsim_base.x -c ./circuits/circuit_q24 -t 4 -f 3 -v 1 -d 12
simu time is 0.131254 seconds.

./apps/qsim_base.x -c ./circuits/circuit_q30 -t 1 -f 3 -v 1 -d 12
simu time is 41.4889 seconds.

./apps/qsim_base.x -c ./circuits/circuit_q30 -t 4 -f 3 -v 1 -d 12
simu time is 11.3796 seconds.
```

Initial observations:

- `q24` proxy scales from `0.46834s` to `0.131254s` going from 1 to 4 threads.
- `q30` proxy scales from `41.4889s` to `11.3796s` going from 1 to 4 threads.
- The printed amplitudes matched between 1-thread and 4-thread runs for each
  circuit, so these outputs can serve as first correctness references.

CPU topology was not recorded in this checkpoint because `sysctl` was blocked
by the current sandbox.

## 2. Instrument the Baseline

Before changing the execution model, measure where the time is going.

Useful instrumentation:

- fuse time
- execution time
- optional counters for how often NEON uses:
  - `ApplyGateH`
  - `ApplyGateL`

This helps answer:

- how much work is already benefiting from low-qubit SIMD-friendly paths
- how much work is still dominated by higher-qubit access patterns

This step is important because tiling helps only if enough useful low-bit-local
work exists to amortize tile scheduling overhead.

### Baseline Timing Split: 2026-06-21

qsim already reports init, fuse, and simulation time with `-v 2`.

Measured proxy runs:

```text
./apps/qsim_base.x -c ./circuits/circuit_q24 -t 4 -f 3 -v 2 -d 12
init time is 9.58971e-07 seconds.
fuse time is 0.000193084 seconds.
simu time is 0.149264 seconds.

./apps/qsim_base.x -c ./circuits/circuit_q30 -t 4 -f 3 -v 2 -d 12
init time is 9.16014e-07 seconds.
fuse time is 0.000171916 seconds.
simu time is 11.4013 seconds.
```

Initial observation:

```text
fusion overhead is tiny compared with state-vector execution time
```

So the first tile experiment should focus on reducing repeated state-vector
traffic, not on reducing fuser overhead.

## 3. Keep the First Tile Experiment Narrow

The first tiled prototype should be intentionally simple:

- define a tile by the lowest `k` physical qubits
- only tile fused gates whose qubits are all within those low `k` bits
- leave all other operations on the original qsim execution path

Suggested prototype control surface:

- `-k tile_low_bits`

At this stage:

- no remapping yet
- no dynamic scheduling
- no attempt to tile every operation

The question is only:

```text
does pure tiled execution help when the gate block is already low-bit-local?
```

## 4. Implement the Minimal Tiled Executor

The minimal tiled path should do:

1. fuse the circuit once
2. scan the fused operations
3. batch consecutive tile-local fused operations
4. execute each batch tile-by-tile
5. fall back to the original simulator path for non-local operations

Recommended execution shape:

```text
fused_ops = FuseGates(circuit)

for op in fused_ops:
    if op is tile-local:
        add to pending block
    else:
        flush pending block tile-by-tile
        run op normally

flush pending block tile-by-tile
```

Recommended threading model:

- outer parallelism over tiles
- single-thread simulator inside each tile

That avoids nested parallelism and keeps the prototype easier to reason about.

## 5. Add Correctness Guardrails

Correctness checks should be added as soon as the tiled path exists.

Suggested checks:

- compare printed amplitudes against the baseline
- compare `q24` and `q30` results
- add one focused small regression test with:
  - a tile-local gate block
  - then a non-local gate
  - then another tile-local gate block

This is mainly to validate that the tiled scheduler flushes correctly at
non-local boundaries.

### Prototype Checkpoint: 2026-06-21

A first narrow prototype was created in a separate worktree:

```text
/Users/xxie/Documents/qsim review/qsim-neon-tile-experiment
branch: codex/neon-tile-experiment
main change: apps/qsim_base.cc adds -k tile_low_bits
```

The prototype keeps the original qsim path as the default. When `-k` is
provided, it:

1. fuses the circuit once
2. batches consecutive fused operations whose qubits are all below `k`
3. applies those batches tile-by-tile
4. falls back to the original full-state path for non-local operations

The threading model is:

```text
parallel over tiles
single-thread simulator inside each tile
```

This avoids nested parallelism. Each tile is represented as a view into the
original state vector using `StateSpace::Create(tile_pointer, tile_qubits)`, so
the prototype does not allocate a second full state vector.

The first correctness smoke check passed for `circuit_q24`:

```text
baseline:
./apps/qsim_base.x -c ./circuits/circuit_q24 -t 4 -f 3 -v 1 -d 12
simu time is 0.134621 seconds.

tiled:
./apps/qsim_base.x -c ./circuits/circuit_q24 -t 4 -f 3 -v 1 -d 12 -k 16
simu time is 0.306413 seconds.
```

The first eight printed amplitudes matched exactly between baseline and tiled
execution. `circuit_q30` also matched for the first eight printed amplitudes
across the sampled tile sizes.

Initial timing samples:

| circuit | command shape | time |
| --- | --- | --- |
| q24 | baseline, `-t 4 -f 3 -d 12` | `0.134621s` |
| q24 | tiled, `-k 14` | `0.163394s` |
| q24 | tiled, `-k 16` | `0.306413s` |
| q24 | tiled, `-k 18` | `0.353661s` |
| q30 | baseline, `-t 4 -f 3 -d 12` | `11.943s` |
| q30 | tiled, `-k 14` | `12.0411s` |
| q30 | tiled, `-k 16` | `11.5209s` |
| q30 | tiled, `-k 18` | `12.4479s` |

Interpretation:

- The prototype is useful as a correctness and measurement harness.
- The current naive tiled scheduler is not reliably faster yet.
- `-k 16` was roughly neutral/slightly faster in one q30 sample, but the margin
  is small enough that repeated trials are needed before claiming a win.
- q24 shows that tile overhead can dominate when the circuit/state is too small
  or when tile-local blocks do not contain enough work.
- The next useful optimization is not remapping yet; it is better tile-local
  block accounting and profitability heuristics.

Recommended next instrumentation:

- count tile-local fused ops
- count non-local barrier ops
- count flushed tile blocks
- count total tile visits, equal to `flushed_blocks * 2^(num_qubits - k)`
- record average tile-local gates per flushed block

This will tell us whether the problem is insufficient locality, too many
barriers, or too little useful work per tile visit.

### Instrumentation and Profitability Checkpoint: 2026-06-21

The prototype now reports tile scheduler stats with `-v 2`:

```text
tile stats:
  k
  tile_qubits
  tiles_per_block
  min_block_ops
  fused_ops
  local_ops
  non_local_ops
  flushed_blocks
  tiled_blocks
  skipped_blocks
  skipped_local_ops
  tile_visits
  avg_block_ops
  max_block_ops
```

It also adds a first profitability knob:

```text
-b tile_min_block_ops
```

If a tile-local block has fewer than `tile_min_block_ops` fused operations, the
prototype skips tiled execution for that block and applies those operations
through the normal full-state qsim path. The default is `-b 1`, which preserves
the original naive tiled behavior.

Measured q30 samples:

| command shape | tile behavior | time |
| --- | --- | --- |
| baseline, no `-k` | original qsim path | `11.53s` |
| `-k 16 -b 1` | tile all 6 local blocks, 98,304 tile visits | `11.4088s` |
| `-k 16 -b 5` | tile 2 blocks, skip 4 blocks, 32,768 tile visits | `11.2942s` |
| `-k 16 -b 7` | tile 1 block, skip 5 blocks, 16,384 tile visits | `11.3376s` |
| `-k 16 -b 9` | tile 0 blocks, skip all local blocks | `11.5035s` |

Measured q24 guardrail samples:

| command shape | tile behavior | time |
| --- | --- | --- |
| `-k 16 -b 5` | tile 2 blocks, skip 3 blocks, 512 tile visits | `0.25112s` |
| `-k 16 -b 9` | tile 0 blocks, skip all local blocks | `0.130777s` |

Interpretation:

- The instrumentation confirms that the naive tiled scheduler creates too much
  work for short local runs.
- q30 sees a small improvement when only larger local blocks are tiled.
- q24 strongly prefers skipping tiling, which is expected for a smaller state
  where the original fused full-state path is already efficient.
- The next scheduler should use a profitability rule before remapping, not
  after remapping.

Practical next heuristic:

```text
tile a block only if:
  block_ops >= threshold
  and estimated saved locality work > tile scheduling overhead
```

The current `-b` knob approximates the first condition. A better next version
should estimate cost using tile count, block size, and whether the block uses
`ApplyGateH`-friendly low qubits inside the tile.

### Snippet-Inspired Layered Extension: 2026-06-21

A second experiment branch was created to extend the snippet-style idea:

```text
branch: codex/neon-tile-experiment-2
main change: apps/qsim_base.cc adds a layered tile-batch runner
```

The raw snippet idea looks like:

```text
for each tile:
  for each fused operation:
    apply operation inside this tile
```

That is only correct when every operation is tile-local. The extended version
keeps the same cache intuition, but inserts an execution-layer builder:

```text
fused_ops = FuseGates(circuit)
layers = BuildExecutionLayers(fused_ops)

for layer in layers:
  if layer is tile-local and profitable:
    for each tile:
      apply all operations in this layer
  else:
    apply the layer through the normal full-state qsim path
```

This preserves the important correctness boundaries:

- tile-local fused gates can be applied tile-by-tile
- non-local gates flush the current tile layer and run on the full state
- measurement and other non-tile-safe operations stay on the qsim path
- tile strides use `StateSpace::MinSize(tile_qubits)`, not raw amplitude count

Initial q24 check:

```text
baseline:
./apps/qsim_base.x -c ./circuits/circuit_q24 -t 4 -f 3 -v 2 -d 12
simu time is 0.131896 seconds.

layered:
./apps/qsim_base.x -c ./circuits/circuit_q24 -t 4 -f 3 -v 2 -d 12 -k 16 -b 5
simu time is 0.247568 seconds.
layer stats: k=16 tile_qubits=16 tiles_per_layer=256 min_layer_ops=5 layers=10 fused_ops=35 local_ops=20 full_state_ops=15 tile_layers=2 full_state_layers=5 skipped_tile_layers=3 skipped_tile_ops=7 tile_visits=512 max_tile_layer_ops=8
```

Initial q30 check:

```text
./apps/qsim_base.x -c ./circuits/circuit_q30 -t 4 -f 3 -v 2 -d 12 -k 16 -b 5
simu time is 11.4859 seconds.
layer stats: k=16 tile_qubits=16 tiles_per_layer=16384 min_layer_ops=5 layers=12 fused_ops=46 local_ops=19 full_state_ops=27 tile_layers=2 full_state_layers=6 skipped_tile_layers=4 skipped_tile_ops=6 tile_visits=32768 max_tile_layer_ops=8
```

The first printed amplitudes matched the baseline checks. Performance is still
not compelling on q24, but the design now captures the missing "layers" concept
from the snippet safely.

The layered branch was then refined to borrow the lean execution style from the
first tiled prototype. Instead of materializing a vector of execution layers
before execution, it now streams through fused operations once:

```text
for op in fused_ops:
  if op is tile-local:
    append op to the current tile layer
  else:
    flush the current tile layer
    apply op immediately on the full state

flush final tile layer
```

This keeps the same layer semantics and stats, but removes the extra
precomputed layer container from the hot path.

Streaming layered check:

```text
./apps/qsim_base.x -c ./circuits/circuit_q24 -t 4 -f 3 -v 2 -d 12 -k 16 -b 5
simu time is 0.258983 seconds.
layer stats: k=16 tile_qubits=16 tiles_per_layer=256 min_layer_ops=5 layers=10 fused_ops=35 local_ops=20 full_state_ops=15 tile_layers=2 full_state_layers=5 skipped_tile_layers=3 skipped_tile_ops=7 tile_visits=512 max_tile_layer_ops=8

./apps/qsim_base.x -c ./circuits/circuit_q30 -t 4 -f 3 -v 2 -d 12 -k 16 -b 5
simu time is 11.336 seconds.
layer stats: k=16 tile_qubits=16 tiles_per_layer=16384 min_layer_ops=5 layers=12 fused_ops=46 local_ops=19 full_state_ops=27 tile_layers=2 full_state_layers=6 skipped_tile_layers=4 skipped_tile_ops=6 tile_visits=32768 max_tile_layer_ops=8
```

The stats match the precomputed-layer version, and the first printed
amplitudes still match the baseline.

### Target Machine Implications: 2026-06-21

The local measurements above were taken on a Mac with an Apple M-series CPU.
The real target is a 112-core machine with smaller L2 caches. This changes the
performance model, but not the correctness model.

What does not change:

- tile-local layers must still be separated from full-state layers
- non-local gates still force a full-state boundary
- measurements and other non-tile-safe operations still need the normal qsim
  path
- tile strides still need `StateSpace::MinSize(tile_qubits)`

What does change:

- the profitable `k` is likely smaller on the target machine
- thread scheduling overhead matters more because there are many more workers
- memory bandwidth and cache contention become the main limit sooner
- a tile size that looks reasonable on the Mac may overflow or pressure the
  smaller target L2

For single-precision state vectors, one tile is roughly:

```text
tile_bytes ~= 2^k complex amplitudes * 8 bytes
           ~= 2^(k + 3) bytes
```

Example tile sizes:

| k | tile bytes | intuition |
| --- | ---: | --- |
| 13 | 64 KiB | safer for small private L2 |
| 14 | 128 KiB | likely first target sweep point |
| 15 | 256 KiB | useful if L2 is at least several hundred KiB |
| 16 | 512 KiB | worked as a local experiment, may be too large |
| 17 | 1 MiB | probably too large for small L2 per core |

For the 112-core machine, the first serious sweep should prioritize:

```text
k = 13, 14, 15, 16
threads = 1, 7, 14, 28, 56, 112
min_layer_ops = 1, 3, 5, 7, 9
```

The key metric is not only runtime. We should also record:

- tile layers
- full-state layers
- tile visits
- average operations per tiled layer
- maximum operations per tiled layer
- speedup from 1 thread to 112 threads

Expected design pressure:

- If small `k` wins, then L2 locality matters more than per-tile overhead.
- If larger `k` wins, then overhead and vector efficiency matter more than L2
  residency.
- If scaling flattens well before 112 threads, the bottleneck is probably
  shared memory bandwidth or full-state barriers, not tile-local compute.

## 6. Benchmark Tile Size Versus Thread Count

Once the tiled path is working, benchmark the interaction between:

- tile size
- thread count
- fusion size

Suggested sweeps:

- `k = 12..18`
- several `-t` values
- one or two representative `-f` values

Questions to answer:

- where does tile reuse begin to help?
- where does tile overhead dominate?
- where does memory bandwidth or shared-cache pressure flatten scaling?

This step is where the connection between tiling and multithreading becomes
visible in practice.

## 7. Evaluate Static Remapping Only After Pure Tiling

After the pure tiled path is understood, test whether static remapping adds
enough value to justify the extra complexity.

At this stage, evaluate:

- whether more fused operations become tile-local after remap
- whether the extra remap work improves end-to-end runtime
- whether the improvement is consistent across both `q24` and `q30`

The question here is:

```text
does remapping increase the fraction of useful tile-local work enough to pay
for itself?
```

## 8. Decide Based on Data

At the end of the experiments, use the measurements to choose the next
direction.

If tiling helps:

- move the prototype toward cleaner helper APIs
- separate the scheduler from `qsim_base`
- improve block selection and heuristics

If tiling does not help enough:

- identify whether the issue is:
  - tile overhead
  - insufficient locality
  - poor tile-local block selection
  - thread/cache contention
  - too little useful work per tile visit

## Recommended Order

The recommended sequence is:

1. baseline measurement
2. pure tiled execution
3. static remap on top of tiling

This order makes it much easier to attribute performance changes to the right
idea.

## Short Summary

The plan is:

```text
measure first
tile second
remap third
clean up only after the data says the idea is worth it
```
