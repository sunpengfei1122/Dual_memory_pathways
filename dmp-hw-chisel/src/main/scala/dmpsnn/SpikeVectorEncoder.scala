package dmpsnn

import chisel3._
import chisel3.util._

/** Converts a dense spike vector into sparse spike addresses.
  *
  * Scans the N-bit spike vector using a priority encoder and outputs
  * up to maxSpikes addresses of neurons that fired. Used between layers
  * to convert layer L's output spike vector into layer L+1's input addresses.
  *
  * Latency: min(numFired, maxSpikes) + 1 cycles.
  */
class SpikeVectorEncoder(
  numNeurons: Int,
  maxSpikes:  Int
) extends Module {
  require(numNeurons > 0)
  require(maxSpikes > 0)

  val io = IO(new Bundle {
    val start    = Input(Bool())
    val done     = Output(Bool())

    val spikeVec = Input(Vec(numNeurons, Bool()))

    val count    = Output(UInt(log2Ceil(maxSpikes + 1).W))
    val addrs    = Output(Vec(maxSpikes, UInt(log2Ceil(numNeurons).W)))
  })

  val sIdle :: sScan :: sDone :: Nil = Enum(3)
  val state = RegInit(sIdle)

  val vecReg   = RegInit(VecInit(Seq.fill(numNeurons)(false.B)))
  val countReg = RegInit(0.U(log2Ceil(maxSpikes + 1).W))
  val addrsReg = RegInit(VecInit(Seq.fill(maxSpikes)(0.U(log2Ceil(numNeurons).W))))

  io.done  := state === sDone
  io.count := countReg
  io.addrs := addrsReg

  switch(state) {
    is(sIdle) {
      when(io.start) {
        vecReg   := io.spikeVec
        countReg := 0.U
        state    := sScan
      }
    }

    is(sScan) {
      val anyActive = vecReg.asUInt.orR
      val lowestIdx = PriorityEncoder(vecReg.asUInt)

      when(anyActive && countReg < maxSpikes.U) {
        addrsReg(countReg) := lowestIdx
        countReg := countReg + 1.U
        vecReg(lowestIdx) := false.B
      }.otherwise {
        state := sDone
      }
    }

    is(sDone) {
      when(io.start) {
        state := sIdle
      }
    }
  }
}
