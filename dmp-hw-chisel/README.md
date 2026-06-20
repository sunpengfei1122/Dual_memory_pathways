# DMP-SNN Chisel Hardware Accelerator

A fully parameterized, synthesizable Chisel 5.x implementation of the Dual Memory Pathway Spiking Neural Network (DMP-SNN) accelerator from:

> **Algorithm-hardware co-design of neuromorphic networks with dual memory pathways**  
> Sun, Su, Achterberg, Indiveri, Goodman, Akarca (2026)  
> arXiv:2512.07602v3

## Architecture

The design implements a digital near-memory-compute architecture with four parallel computation paths per timestep:

```
                    ┌─── SpikeIntegration (Wf·s)     ──── input-stationary ───┐
Input Spikes ───→  ├─── ScalarDrive (Wx·s + b → x[k]) ──────────────────────┼──→ NeuronBank ──→ Output Spikes
  (AER)             ├─── MemoryIntegration (P·m[k-1] + v·x[k]) ── output-stat ┤     (fused LIF)
                    └─── MemoryUpdate (Ā·m[k-1] + B̄·x[k]) ──── parallel ────┘
```

### Key Hardware Optimizations 
1. **Dataflow-dependency breaking** — `P·m[k-1]` starts immediately using the previous memory state; `v·x[k]` starts once the scalar drive completes. Both run in parallel with spike integration.

2. **Operator fusion** — The neuron bank fuses leak, spike integration, memory integration, threshold comparison, and reset into a single SRAM read/write per neuron group, maximizing arithmetic intensity.

3. **Heterogeneous operand stationarity** — Input-stationary access for sparse spike integration (minimizes weight-SRAM reads); output-stationary for dense memory integration (minimizes neuron-SRAM reads).

## Configuration

All parameters are set via `DmpConfig`:

```scala
DmpConfig(
  nNeurons        = 128,   // Hidden layer width (N)
  memDim          = 8,     // Slow memory dimension (d << N)
  inputWidth      = 700,   // Input spike channels (SHD = 700)
  maxSpikesPerStep = 32,   // Sparsity bound per timestep
  wBits           = 8,     // Weight precision
  uBits           = 16,    // Membrane potential precision
  mBits           = 16,    // Memory state precision
  accBits         = 24,    // Accumulator width
  beta            = 230,   // Leak factor (230/256 ≈ 0.898)
  threshold       = 512,   // Firing threshold
  fusedNeurons    = 4      // Neurons processed per cycle (register slots)
)
```

## Project Structure

```
src/main/scala/dmpsnn/
├── package.scala              Constants (BETA_SHIFT)
├── DmpConfig.scala            Configuration case class
├── DmpNetworkConfig.scala     Multi-layer network configuration
├── SramInterface.scala        Sram / WideSram primitives
├── SpikeIntegration.scala     Path 1: sparse input-stationary Wf·s
├── ScalarDrive.scala          Scalar x[k] = Wx·s + b
├── MemoryUpdate.scala         Path 4: m[k] = Ā·m[k-1] + B̄·x[k]
├── MemoryIntegration.scala    Paths 2+3: output-stationary P·m[k-1] + v·x[k]
├── NeuronBank.scala           Fused LIF (leak + accumulate + threshold + reset)
├── SpikeVectorEncoder.scala   Dense-to-sparse spike conversion (inter-layer)
├── DmpCore.scala              Single-layer FSM orchestrating all parallel datapaths
├── DmpTop.scala               Single-layer top with AER I/O and weight loading
└── DmpMultiLayerTop.scala     Multi-layer top with sequential layer execution

src/test/scala/dmpsnn/
├── SpikeIntegrationSpec.scala
├── ScalarDriveSpec.scala
├── MemoryUpdateSpec.scala
├── NeuronBankSpec.scala
├── SpikeVectorEncoderSpec.scala
├── DmpCoreSpec.scala
└── DmpMultiLayerSpec.scala
```

## Prerequisites

- Java 11+
- sbt 1.9+

## Build & Test

```bash
# Compile all modules
sbt compile

# Run tests
sbt test

# Generate SystemVerilog (single-layer)
sbt "runMain dmpsnn.DmpTopVerilog"

# Generate SystemVerilog (multi-layer, SHD configuration)
sbt "runMain dmpsnn.DmpMultiLayerVerilog"
```

## Computation per Timestep

| Path | Operation | Dataflow | Cycles |
|------|-----------|----------|--------|
| 1 | Wf · s (sparse) | Input-stationary | maxSpikes × N/fusedNeurons |
| 2 | P · m[k-1] (dense) | Output-stationary | (N/fusedNeurons) × d |
| 3 | v · x[k] (dense) | Output-stationary | N/fusedNeurons |
| 4 | Ā · m[k-1] + B̄·x[k] | Sequential | d + 1 |
| Fuse | Leak + accumulate + fire | Streaming | N/fusedNeurons × 3 |

Paths 1, 2, and 4 execute in parallel. Path 3 starts after ScalarDrive completes. The NeuronBank begins once Paths 1 and 2+3 finish.

## References

- Paper: [arXiv:2512.07602v3](https://arxiv.org/abs/2512.07602v3)
- Chisel: [chisel-lang.org](https://www.chisel-lang.org/)
- Legendre Memory Unit: Voelker et al., NeurIPS 2019

## License

Research use. See paper for original architecture IP.
