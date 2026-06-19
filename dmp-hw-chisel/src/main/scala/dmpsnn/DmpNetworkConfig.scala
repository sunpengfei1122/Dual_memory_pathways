package dmpsnn

import chisel3.util._

case class DmpNetworkConfig(
  layers: Seq[DmpConfig],
  maxSpikesInterLayer: Int = 32
) {
  require(layers.nonEmpty, "At least one layer required")

  for (i <- 0 until layers.size - 1) {
    require(
      layers(i).nNeurons == layers(i + 1).inputWidth,
      s"Layer $i nNeurons (${layers(i).nNeurons}) must equal layer ${i + 1} inputWidth (${layers(i + 1).inputWidth})"
    )
  }

  for (i <- 1 until layers.size) {
    require(
      maxSpikesInterLayer <= layers(i).maxSpikesPerStep,
      s"maxSpikesInterLayer ($maxSpikesInterLayer) must <= layer $i maxSpikesPerStep (${layers(i).maxSpikesPerStep})"
    )
  }

  val numLayers: Int = layers.size
  val outputNeurons: Int = layers.last.nNeurons
  val maxInputSpikes: Int = layers.head.maxSpikesPerStep
  val maxSpikeAddrBits: Int = layers.map(_.spikeAddrBits).max
  val maxNeuronAddrBits: Int = layers.map(_.neuronAddrBits).max
  val maxMaxSpikes: Int = layers.map(_.maxSpikesPerStep).max
}
