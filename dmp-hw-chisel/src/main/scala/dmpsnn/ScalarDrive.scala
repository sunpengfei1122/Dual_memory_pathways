package dmpsnn

import chisel3._
import chisel3.util._

/** Computes the scalar drive x[k] = Wx · s[k-1] + b
  *
  * Iterates over active spike addresses, accumulating Wx[addr] for each.
  * Runs concurrently with SpikeIntegration using the same spike address stream.
  */
class ScalarDrive(cfg: DmpConfig) extends Module {
  val io = IO(new Bundle {
    val start      = Input(Bool())
    val done       = Output(Bool())

    // Spike input (shared with SpikeIntegration)
    val spikeCount = Input(UInt(log2Ceil(cfg.maxSpikesPerStep + 1).W))
    val spikeAddr  = Input(UInt(cfg.spikeAddrBits.W))
    val spikeIdx   = Output(UInt(log2Ceil(cfg.maxSpikesPerStep + 1).W))

    // Wx weight SRAM (inputWidth entries, wBits each)
    val wxSram = new Bundle {
      val en   = Output(Bool())
      val addr = Output(UInt(cfg.spikeAddrBits.W))
      val rdata = Input(SInt(cfg.wBits.W))
    }

    // Bias (loaded from config registers)
    val bias = Input(SInt(cfg.mBits.W))

    // Output scalar x[k]
    val xOut      = Output(SInt(cfg.mBits.W))
    val xOutValid = Output(Bool())
  })

  val sIdle :: sProcess :: sDone :: Nil = Enum(3)
  val state = RegInit(sIdle)

  val acc = RegInit(0.S(cfg.accBits.W))
  val spikeCounter = RegInit(0.U(log2Ceil(cfg.maxSpikesPerStep + 1).W))
  val validPipe = RegInit(false.B)

  io.done := state === sDone && !validPipe
  io.spikeIdx := spikeCounter
  io.wxSram.en := false.B
  io.wxSram.addr := 0.U
  io.xOutValid := state === sDone && !validPipe

  // ReLU then saturate: x[k] = fx(Wx·s + b) where fx = ReLU (per paper Eq.7, Fig.4a)
  val xSum = acc + io.bias.pad(cfg.accBits)
  val xMax = ((1L << (cfg.mBits - 1)) - 1).S(cfg.accBits.W)
  when(xSum <= 0.S) {
    io.xOut := 0.S(cfg.mBits.W)
  }.elsewhen(xSum > xMax) {
    io.xOut := ((1L << (cfg.mBits - 1)) - 1).S(cfg.mBits.W)
  }.otherwise {
    io.xOut := xSum(cfg.mBits - 1, 0).asSInt
  }

  switch(state) {
    is(sIdle) {
      when(io.start) {
        acc := 0.S
        spikeCounter := 0.U
        validPipe := false.B
        state := Mux(io.spikeCount === 0.U, sDone, sProcess)
      }
    }

    is(sProcess) {
      io.wxSram.en := true.B
      io.wxSram.addr := io.spikeAddr

      when(validPipe) {
        acc := acc + io.wxSram.rdata.pad(cfg.accBits)
      }
      validPipe := true.B

      when(spikeCounter === io.spikeCount - 1.U) {
        state := sDone
      }.otherwise {
        spikeCounter := spikeCounter + 1.U
      }
    }

    is(sDone) {
      when(validPipe) {
        acc := acc + io.wxSram.rdata.pad(cfg.accBits)
        validPipe := false.B
      }
      when(io.start) {
        acc := 0.S
        spikeCounter := 0.U
        validPipe := false.B
        state := Mux(io.spikeCount === 0.U, sDone, sProcess)
      }
    }
  }
}
