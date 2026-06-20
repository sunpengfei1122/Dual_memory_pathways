package dmpsnn

import chisel3._
import chisel3.util._

/** Path 4: Slow memory state update.
  *
  * Computes: m[k] = Ā · m[k-1] + B̄ · x[k]
  *
  * - Ā is a d×d state-transition matrix (Legendre-derived, fixed after discretization)
  * - B̄ is a d×1 input vector (fixed)
  * - m is stored in a register file (d is small, typically 5-16)
  *
  * The update takes d+1 cycles: d cycles for the matrix-vector product Ā·m[k-1],
  * plus 1 cycle for adding B̄·x[k].
  */
class MemoryUpdate(cfg: DmpConfig) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val done  = Output(Bool())

    // Scalar drive input (from ScalarDrive module)
    val xIn      = Input(SInt(cfg.mBits.W))
    val xInValid = Input(Bool())

    // Current memory state (exposed for MemoryIntegration to read m[k-1])
    val mState    = Output(Vec(cfg.memDim, SInt(cfg.mBits.W)))
    // Previous memory state (stable during computation, used by MemInteg path A)
    val mStatePrev = Output(Vec(cfg.memDim, SInt(cfg.mBits.W)))

    // Ā matrix SRAM: d rows × d cols, stored row-major
    val aBarSram = new Bundle {
      val en    = Output(Bool())
      val addr  = Output(UInt(log2Ceil(cfg.memDim).W))
      val rdata = Input(Vec(cfg.memDim, SInt(cfg.wBits.W)))
    }

    // B̄ vector (stored in registers, loaded at init)
    val bBar = Input(Vec(cfg.memDim, SInt(cfg.wBits.W)))
  })

  // Memory state register files
  val mCurr = RegInit(VecInit(Seq.fill(cfg.memDim)(0.S(cfg.mBits.W))))
  val mPrev = RegInit(VecInit(Seq.fill(cfg.memDim)(0.S(cfg.mBits.W))))

  // Accumulator for new m[k] (widened to prevent dot product overflow)
  val mNext = RegInit(VecInit(Seq.fill(cfg.memDim)(0.S(cfg.internalAccBits.W))))

  // FSM
  val sIdle :: sMatVec :: sWaitX :: sAddBx :: sDone :: Nil = Enum(5)
  val state = RegInit(sIdle)

  val rowCounter = RegInit(0.U(log2Ceil(cfg.memDim + 1).W))
  val validPipe  = RegInit(false.B)
  val rowDelayed = RegNext(rowCounter)

  io.done := state === sDone
  io.mState := mCurr
  io.mStatePrev := mPrev
  io.aBarSram.en := false.B
  io.aBarSram.addr := 0.U

  switch(state) {
    is(sIdle) {
      when(io.start) {
        // Snapshot current state as "previous" for MemoryIntegration
        for (i <- 0 until cfg.memDim) {
          mPrev(i) := mCurr(i)
          mNext(i) := 0.S
        }
        rowCounter := 0.U
        validPipe := false.B
        state := sMatVec
      }
    }

    is(sMatVec) {
      // Compute Ā · m[k-1]: one row of Ā per cycle
      // Read row `rowCounter` of Ā
      io.aBarSram.en := true.B
      io.aBarSram.addr := rowCounter

      // Accumulate dot product from previous row read
      when(validPipe) {
        val terms = (0 until cfg.memDim).map { j =>
          (io.aBarSram.rdata(j) * mPrev(j)).pad(cfg.internalAccBits)
        }
        mNext(rowDelayed) := terms.reduce(_ + _)
      }
      validPipe := true.B

      when(rowCounter === (cfg.memDim - 1).U) {
        state := sWaitX
      }.otherwise {
        rowCounter := rowCounter + 1.U
      }
    }

    is(sWaitX) {
      // Process last pipelined row
      when(validPipe) {
        val terms = (0 until cfg.memDim).map { j =>
          (io.aBarSram.rdata(j) * mPrev(j)).pad(cfg.internalAccBits)
        }
        mNext(rowDelayed) := terms.reduce(_ + _)
        validPipe := false.B
      }
      // Wait for x[k] to be available
      when(io.xInValid) {
        state := sAddBx
      }
    }

    is(sAddBx) {
      // Add B̄ · x[k] to each component with saturation
      val mMax = ((1L << (cfg.mBits - 1)) - 1).S(cfg.internalAccBits.W)
      val mMin = (-(1L << (cfg.mBits - 1))).S(cfg.internalAccBits.W)
      for (i <- 0 until cfg.memDim) {
        val bx = (io.bBar(i) * io.xIn).pad(cfg.internalAccBits)
        val newVal = mNext(i) + bx
        // Saturate to mBits
        when(newVal > mMax) {
          mCurr(i) := ((1L << (cfg.mBits - 1)) - 1).S(cfg.mBits.W)
        }.elsewhen(newVal < mMin) {
          mCurr(i) := (-(1L << (cfg.mBits - 1))).S(cfg.mBits.W)
        }.otherwise {
          mCurr(i) := newVal(cfg.mBits - 1, 0).asSInt
        }
      }
      state := sDone
    }

    is(sDone) {
      when(io.start) {
        for (i <- 0 until cfg.memDim) {
          mPrev(i) := mCurr(i)
          mNext(i) := 0.S
        }
        rowCounter := 0.U
        validPipe := false.B
        state := sMatVec
      }
    }
  }
}
