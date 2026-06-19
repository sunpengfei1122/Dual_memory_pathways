package dmpsnn

import chisel3._
import chisel3.util._

/** Fused LIF Neuron Bank — the key operator-fusion optimization.
  *
  * Fuses leak + spike_integration + memory_integration + threshold + reset
  * into a single pass over neuron SRAM with one read and one write per group.
  *
  * Processes `fusedNeurons` neurons per cycle using temporary register slots:
  *   1. Read u[i:i+3] from neuron SRAM
  *   2. Apply leak: u_leaked = (β * u_old) >> BETA_SHIFT
  *   3. Add I_spike[i:i+3]
  *   4. Add I_mem[i:i+3]
  *   5. Threshold comparison → generate spikes
  *   6. Reset fired neurons (subtract threshold)
  *   7. Write u_new[i:i+3] back to SRAM
  */
class NeuronBank(cfg: DmpConfig) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val done  = Output(Bool())

    // Inputs from integration paths (full vectors, pre-computed)
    val iSpike = Input(Vec(cfg.nNeurons, SInt(cfg.accBits.W)))
    val iMem   = Input(Vec(cfg.nNeurons, SInt(cfg.accBits.W)))

    // Neuron state SRAM (stores membrane potentials)
    val neuronSram = new Bundle {
      val rEn   = Output(Bool())
      val rAddr = Output(UInt(log2Ceil(cfg.neuronGroups).W))
      val rData = Input(Vec(cfg.fusedNeurons, SInt(cfg.uBits.W)))
      val wEn   = Output(Bool())
      val wAddr = Output(UInt(log2Ceil(cfg.neuronGroups).W))
      val wData = Output(Vec(cfg.fusedNeurons, SInt(cfg.uBits.W)))
    }

    // Output spikes for this timestep
    val spikesOut = Output(Vec(cfg.nNeurons, Bool()))
    val spikesValid = Output(Bool())

    // Membrane potential readout (for output layer classification)
    val membraneOut = Output(Vec(cfg.nNeurons, SInt(cfg.uBits.W)))
  })

  val sIdle :: sRead :: sCompute :: sWrite :: sDone :: Nil = Enum(5)
  val state = RegInit(sIdle)

  val groupIdx = RegInit(0.U(log2Ceil(cfg.neuronGroups + 1).W))
  val groupIdxDelayed = RegNext(groupIdx)

  // Spike output register
  val spikes = RegInit(VecInit(Seq.fill(cfg.nNeurons)(false.B)))

  // Membrane potential readout register
  val membraneReg = RegInit(VecInit(Seq.fill(cfg.nNeurons)(0.S(cfg.uBits.W))))

  // Temporary register slots for fused computation (4 neurons)
  val uRegs = Reg(Vec(cfg.fusedNeurons, SInt(cfg.uBits.W)))

  io.done := state === sDone
  io.spikesOut := spikes
  io.spikesValid := state === sDone
  io.membraneOut := membraneReg

  io.neuronSram.rEn := false.B
  io.neuronSram.rAddr := 0.U
  io.neuronSram.wEn := false.B
  io.neuronSram.wAddr := 0.U
  io.neuronSram.wData := VecInit(Seq.fill(cfg.fusedNeurons)(0.S(cfg.uBits.W)))

  switch(state) {
    is(sIdle) {
      when(io.start) {
        groupIdx := 0.U
        state := sRead
      }
    }

    is(sRead) {
      // Issue SRAM read for current neuron group
      io.neuronSram.rEn := true.B
      io.neuronSram.rAddr := groupIdx
      state := sCompute
    }

    is(sCompute) {
      // Fused computation on the 4 neurons read from SRAM
      val compWidth = cfg.uBits + BETA_SHIFT + 2  // working precision
      for (i <- 0 until cfg.fusedNeurons) {
        val neuronIdx = groupIdxDelayed * cfg.fusedNeurons.U + i.U
        val uOld = io.neuronSram.rData(i)

        // Step 1: Leak — (β * u) >> BETA_SHIFT (fixed-point multiply)
        val product = Wire(SInt((cfg.uBits + 9).W))
        product := uOld * cfg.beta.S(9.W)
        val leaked = (product >> BETA_SHIFT).asSInt

        // Step 2+3: Add integration currents (sign-extended to working width)
        val iSpikeW = io.iSpike(neuronIdx)(cfg.uBits - 1, 0).asSInt
        val iMemW = io.iMem(neuronIdx)(cfg.uBits - 1, 0).asSInt

        // Step 4: Accumulate in wider precision
        val uNew = Wire(SInt(compWidth.W))
        uNew := leaked.pad(compWidth) + iSpikeW.pad(compWidth) + iMemW.pad(compWidth)

        // Step 5: Threshold comparison
        val threshWire = cfg.threshold.S(compWidth.W)
        val fired = uNew >= threshWire

        // Step 6: Reset (subtract threshold if fired)
        val uReset = Mux(fired, uNew - threshWire, uNew)

        // Saturate to uBits
        val maxVal = ((1 << (cfg.uBits - 1)) - 1).S(compWidth.W)
        val minVal = (-(1 << (cfg.uBits - 1))).S(compWidth.W)
        val uSat = Wire(SInt(cfg.uBits.W))
        when(uReset > maxVal) {
          uSat := ((1 << (cfg.uBits - 1)) - 1).S(cfg.uBits.W)
        }.elsewhen(uReset < minVal) {
          uSat := (-(1 << (cfg.uBits - 1))).S(cfg.uBits.W)
        }.otherwise {
          uSat := uReset(cfg.uBits - 1, 0).asSInt
        }

        uRegs(i) := uSat
        membraneReg(neuronIdx) := uSat
        spikes(neuronIdx) := fired
      }
      state := sWrite
    }

    is(sWrite) {
      // Write updated membrane potentials back to SRAM
      io.neuronSram.wEn := true.B
      io.neuronSram.wAddr := groupIdxDelayed
      io.neuronSram.wData := uRegs

      when(groupIdxDelayed === (cfg.neuronGroups - 1).U) {
        state := sDone
      }.otherwise {
        // Advance to next group — read in next cycle (sRead)
        groupIdx := groupIdxDelayed + 1.U
        state := sRead
      }
    }

    is(sDone) {
      when(io.start) {
        state := sIdle
      }
    }
  }
}
