package dmpsnn

import chisel3._
import chisel3.util._

case class DmpConfig(
  nNeurons:        Int = 128,
  memDim:          Int = 8,
  inputWidth:      Int = 700,
  maxSpikesPerStep: Int = 32,
  wBits:           Int = 8,
  uBits:           Int = 16,
  mBits:           Int = 16,
  accBits:         Int = 24,
  beta:            Int = 230,   // fixed-point β: 230/256 ≈ 0.898
  threshold:       Int = 512,
  fusedNeurons:    Int = 4
) {
  require(nNeurons % fusedNeurons == 0, "nNeurons must be divisible by fusedNeurons")
  require(memDim > 0 && memDim <= nNeurons, "memDim must be in (0, nNeurons]")

  val neuronGroups: Int = nNeurons / fusedNeurons
  val spikeAddrBits: Int = log2Ceil(inputWidth)
  val neuronAddrBits: Int = log2Ceil(nNeurons)
  val memAddrBits: Int = log2Ceil(memDim)
}
