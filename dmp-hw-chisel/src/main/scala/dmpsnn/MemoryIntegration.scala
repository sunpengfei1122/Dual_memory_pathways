package dmpsnn

import chisel3._
import chisel3.util._

/** Paths 2+3: Memory integration with output-stationary dataflow.
  *
  * Computes: I_mem[k] = P · m[k-1] + v · x[k]
  *
  * Dependency breaking (from paper Section 2.4.1):
  * - Path 2 (P·m[k-1]) starts immediately using the previous memory state
  * - Path 3 (v·x[k]) starts once the scalar drive x[k] is available
  * - Both are accumulated output-stationary: preload fusedNeurons accumulators,
  *   stream d weights per neuron group
  *
  * P is N×d (stored row-major, fusedNeurons rows per SRAM word)
  * v is N×1
  */
class MemoryIntegration(cfg: DmpConfig) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val done  = Output(Bool())

    // Previous memory state (from MemoryUpdate, stable during computation)
    val mPrev = Input(Vec(cfg.memDim, SInt(cfg.mBits.W)))

    // Scalar drive (from ScalarDrive)
    val xIn      = Input(SInt(cfg.mBits.W))
    val xInValid = Input(Bool())

    // P matrix SRAM: N×d, stored as neuronGroups rows of (fusedNeurons × d) blocks
    // Each read returns fusedNeurons weights for one memory dimension
    val pSram = new Bundle {
      val en   = Output(Bool())
      val addr = Output(UInt(log2Ceil(cfg.neuronGroups * cfg.memDim).W))
      val rdata = Input(Vec(cfg.fusedNeurons, SInt(cfg.wBits.W)))
    }

    // v vector SRAM: N entries
    val vSram = new Bundle {
      val en   = Output(Bool())
      val addr = Output(UInt(log2Ceil(cfg.neuronGroups).W))
      val rdata = Input(Vec(cfg.fusedNeurons, SInt(cfg.wBits.W)))
    }

    // Output: memory-integration current for each neuron
    val result    = Output(Vec(cfg.nNeurons, SInt(cfg.accBits.W)))
    val resultValid = Output(Bool())
  })

  // Accumulators for output-stationary computation (widened to prevent overflow)
  val acc = RegInit(VecInit(Seq.fill(cfg.nNeurons)(0.S(cfg.internalAccBits.W))))

  // FSM
  val sIdle :: sPathA :: sPathB :: sDone :: Nil = Enum(4)
  val state = RegInit(sIdle)

  // Path A: P·m[k-1] — iterate over neuron groups, for each group iterate over memDim
  val groupCounter = RegInit(0.U(log2Ceil(cfg.neuronGroups + 1).W))
  val dimCounter   = RegInit(0.U(log2Ceil(cfg.memDim + 1).W))
  val validPipeA   = RegInit(false.B)
  val groupDelayedA = RegNext(groupCounter)
  val dimDelayedA   = RegNext(dimCounter)

  // Path B: v·x[k]
  val groupCounterB = RegInit(0.U(log2Ceil(cfg.neuronGroups + 1).W))
  val validPipeB    = RegInit(false.B)
  val groupDelayedB = RegNext(groupCounterB)
  val xLatched      = RegInit(0.S(cfg.mBits.W))

  io.done := state === sDone && !validPipeB
  io.pSram.en := false.B
  io.pSram.addr := 0.U
  io.vSram.en := false.B
  io.vSram.addr := 0.U
  io.resultValid := state === sDone && !validPipeB

  // Saturate internal accumulator to accBits output width
  val accMax = ((1L << (cfg.accBits - 1)) - 1).S(cfg.internalAccBits.W)
  val accMin = (-(1L << (cfg.accBits - 1))).S(cfg.internalAccBits.W)
  for (i <- 0 until cfg.nNeurons) {
    when(acc(i) > accMax) {
      io.result(i) := ((1L << (cfg.accBits - 1)) - 1).S(cfg.accBits.W)
    }.elsewhen(acc(i) < accMin) {
      io.result(i) := (-(1L << (cfg.accBits - 1))).S(cfg.accBits.W)
    }.otherwise {
      io.result(i) := acc(i)(cfg.accBits - 1, 0).asSInt
    }
  }

  switch(state) {
    is(sIdle) {
      when(io.start) {
        for (i <- 0 until cfg.nNeurons) {
          acc(i) := 0.S
        }
        groupCounter := 0.U
        dimCounter := 0.U
        validPipeA := false.B
        validPipeB := false.B
        groupCounterB := 0.U
        state := sPathA
      }
    }

    is(sPathA) {
      // Output-stationary: for each neuron group, accumulate contributions from all d memory dims
      io.pSram.en := true.B
      io.pSram.addr := groupCounter * cfg.memDim.U + dimCounter

      // Accumulate previous read
      when(validPipeA) {
        for (i <- 0 until cfg.fusedNeurons) {
          val neuronIdx = groupDelayedA * cfg.fusedNeurons.U + i.U
          val contribution = (io.pSram.rdata(i) * io.mPrev(dimDelayedA)).pad(cfg.internalAccBits)
          acc(neuronIdx) := acc(neuronIdx) + contribution
        }
      }
      validPipeA := true.B

      // Advance counters: inner loop over memDim, outer over neuronGroups
      when(dimCounter === (cfg.memDim - 1).U) {
        dimCounter := 0.U
        when(groupCounter === (cfg.neuronGroups - 1).U) {
          // Path A complete, move to Path B (wait for x if needed)
          state := sPathB
        }.otherwise {
          groupCounter := groupCounter + 1.U
        }
      }.otherwise {
        dimCounter := dimCounter + 1.U
      }
    }

    is(sPathB) {
      // Drain last Path A pipeline stage
      when(validPipeA) {
        for (i <- 0 until cfg.fusedNeurons) {
          val neuronIdx = groupDelayedA * cfg.fusedNeurons.U + i.U
          val contribution = (io.pSram.rdata(i) * io.mPrev(dimDelayedA)).pad(cfg.internalAccBits)
          acc(neuronIdx) := acc(neuronIdx) + contribution
        }
        validPipeA := false.B
      }

      // Path B: v · x[k] — needs x to be valid
      when(io.xInValid) {
        when(!validPipeB && groupCounterB === 0.U) {
          // First cycle: latch x and start reading v
          xLatched := io.xIn
        }

        io.vSram.en := true.B
        io.vSram.addr := groupCounterB

        // Accumulate v[i] * x for previous group
        when(validPipeB) {
          for (i <- 0 until cfg.fusedNeurons) {
            val neuronIdx = groupDelayedB * cfg.fusedNeurons.U + i.U
            val contribution = (io.vSram.rdata(i) * xLatched).pad(cfg.internalAccBits)
            acc(neuronIdx) := acc(neuronIdx) + contribution
          }
        }
        validPipeB := true.B

        when(groupCounterB === (cfg.neuronGroups - 1).U) {
          state := sDone
        }.otherwise {
          groupCounterB := groupCounterB + 1.U
        }
      }
    }

    is(sDone) {
      // Drain last Path B pipeline stage
      when(validPipeB) {
        for (i <- 0 until cfg.fusedNeurons) {
          val neuronIdx = groupDelayedB * cfg.fusedNeurons.U + i.U
          val contribution = (io.vSram.rdata(i) * xLatched).pad(cfg.internalAccBits)
          acc(neuronIdx) := acc(neuronIdx) + contribution
        }
        validPipeB := false.B
      }
      when(io.start) {
        for (i <- 0 until cfg.nNeurons) {
          acc(i) := 0.S
        }
        groupCounter := 0.U
        dimCounter := 0.U
        validPipeA := false.B
        validPipeB := false.B
        groupCounterB := 0.U
        state := sPathA
      }
    }
  }
}
