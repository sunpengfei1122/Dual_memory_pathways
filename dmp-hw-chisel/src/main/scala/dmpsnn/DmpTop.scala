package dmpsnn

import chisel3._
import chisel3.util._

/** AER (Address-Event Representation) event packet. */
class AerEvent(addrWidth: Int) extends Bundle {
  val addr = UInt(addrWidth.W)
}

/** Weight loading interface for initializing SRAMs. */
class WeightLoadPort(cfg: DmpConfig) extends Bundle {
  val valid = Input(Bool())
  val ready = Output(Bool())
  val target = Input(UInt(3.W))  // 0=Wf, 1=Wx, 2=P, 3=v, 4=Abar, 5=bBar, 6=bias
  val addr  = Input(UInt(16.W))
  val data  = Input(UInt((math.max(cfg.fusedNeurons, cfg.memDim) * cfg.wBits).W))
}

/** DMP-SNN Top-level module with AER interface.
  *
  * Provides:
  * - AER spike input (streaming address events per timestep)
  * - AER spike output (address events for downstream)
  * - Weight/parameter loading interface
  * - Optional mean membrane readout for classification
  */
class DmpTop(cfg: DmpConfig) extends Module {
  val io = IO(new Bundle {
    // AER input: spike events for current timestep
    val aerIn = new Bundle {
      val valid = Input(Bool())
      val ready = Output(Bool())
      val event = Input(new AerEvent(cfg.spikeAddrBits))
      val last  = Input(Bool())  // marks end of current timestep's events
    }

    // AER output: spike events produced by the layer
    val aerOut = new Bundle {
      val valid = Output(Bool())
      val ready = Input(Bool())
      val event = Output(new AerEvent(cfg.neuronAddrBits))
      val last  = Output(Bool())
    }

    // Mean membrane readout (for classification at output layer)
    val membraneOut = new Bundle {
      val valid = Output(Bool())
      val data  = Output(Vec(cfg.nNeurons, SInt(cfg.uBits.W)))
    }

    // Weight loading
    val weightLoad = new WeightLoadPort(cfg)

    // Status
    val busy     = Output(Bool())
    val timestep = Output(UInt(32.W))
  })

  val core = Module(new DmpCore(cfg))

  // ===== AER Input: buffer spike addresses until `last` =====
  val sIdle :: sCollect :: sProcess :: sOutputSpikes :: Nil = Enum(4)
  val state = RegInit(sIdle)

  val spikeBuffer = RegInit(VecInit(Seq.fill(cfg.maxSpikesPerStep)(0.U(cfg.spikeAddrBits.W))))
  val spikeCount  = RegInit(0.U(log2Ceil(cfg.maxSpikesPerStep + 1).W))

  // Output spike encoder
  val outSpikeIdx = RegInit(0.U(log2Ceil(cfg.nNeurons + 1).W))
  val outSpikesReg = RegInit(VecInit(Seq.fill(cfg.nNeurons)(false.B)))

  io.aerIn.ready := state === sIdle || state === sCollect
  io.aerOut.valid := false.B
  io.aerOut.event.addr := 0.U
  io.aerOut.last := false.B
  io.membraneOut.valid := false.B
  io.membraneOut.data := VecInit(Seq.fill(cfg.nNeurons)(0.S(cfg.uBits.W)))
  io.busy := state =/= sIdle
  io.timestep := core.io.timestep

  // Weight load — pass through to core
  core.io.weightLoad.valid  := io.weightLoad.valid
  core.io.weightLoad.target := io.weightLoad.target
  core.io.weightLoad.addr   := io.weightLoad.addr
  core.io.weightLoad.data   := io.weightLoad.data
  io.weightLoad.ready := core.io.weightLoad.ready

  // Core connections
  core.io.inSpikes.valid := false.B
  core.io.inSpikes.count := spikeCount
  core.io.inSpikes.addrs := spikeBuffer

  switch(state) {
    is(sIdle) {
      spikeCount := 0.U
      when(io.aerIn.valid) {
        spikeBuffer(0) := io.aerIn.event.addr
        spikeCount := 1.U
        state := Mux(io.aerIn.last, sProcess, sCollect)
      }
    }

    is(sCollect) {
      when(io.aerIn.valid) {
        when(spikeCount < cfg.maxSpikesPerStep.U) {
          spikeBuffer(spikeCount) := io.aerIn.event.addr
          spikeCount := spikeCount + 1.U
        }
        when(io.aerIn.last) {
          state := sProcess
        }
      }
    }

    is(sProcess) {
      // Launch core computation
      core.io.inSpikes.valid := true.B
      core.io.inSpikes.count := spikeCount

      when(core.io.outSpikes.valid) {
        // Latch output spikes for AER encoding
        outSpikesReg := core.io.outSpikes.vector
        outSpikeIdx := 0.U
        state := sOutputSpikes
      }
    }

    is(sOutputSpikes) {
      // Scan through output neurons and emit AER events for those that fired
      val foundSpike = Wire(Bool())
      val nextIdx = Wire(UInt(log2Ceil(cfg.nNeurons + 1).W))
      foundSpike := false.B
      nextIdx := outSpikeIdx

      // Find next active spike from current index
      for (i <- cfg.nNeurons - 1 to 0 by -1) {
        when(i.U >= outSpikeIdx && outSpikesReg(i)) {
          foundSpike := true.B
          nextIdx := i.U
        }
      }

      when(foundSpike) {
        io.aerOut.valid := true.B
        io.aerOut.event.addr := nextIdx

        when(io.aerOut.ready) {
          outSpikesReg(nextIdx) := false.B
          outSpikeIdx := nextIdx + 1.U

          // Check if this was the last spike
          val anyRemaining = outSpikesReg.zipWithIndex.map { case (s, i) =>
            s && i.U > nextIdx
          }.reduce(_ || _)
          when(!anyRemaining) {
            io.aerOut.last := true.B
            state := sIdle
          }
        }
      }.otherwise {
        // No spikes fired this timestep
        io.aerOut.valid := true.B
        io.aerOut.event.addr := 0.U
        io.aerOut.last := true.B
        when(io.aerOut.ready) {
          state := sIdle
        }
      }
    }
  }
}

/** Generate SystemVerilog for the single-layer DMP-SNN top module. */
object DmpTopVerilog extends App {
  val cfg = DmpConfig(
    nNeurons = 128,
    memDim = 8,
    inputWidth = 700,
    maxSpikesPerStep = 32,
    fusedNeurons = 4
  )
  chisel3.emitVerilog(new DmpTop(cfg), Array("--target-dir", "generated"))
}

/** Generate SystemVerilog for the multi-layer DMP-SNN (2-layer SHD config). */
object DmpMultiLayerVerilog extends App {
  val netCfg = DmpNetworkConfig(layers = Seq(
    DmpConfig(
      nNeurons = 128, memDim = 8, inputWidth = 700,
      maxSpikesPerStep = 32, fusedNeurons = 4
    ),
    DmpConfig(
      nNeurons = 20, memDim = 4, inputWidth = 128,
      maxSpikesPerStep = 32, fusedNeurons = 2
    )
  ))
  chisel3.emitVerilog(new DmpMultiLayerTop(netCfg), Array("--target-dir", "generated"))
}
