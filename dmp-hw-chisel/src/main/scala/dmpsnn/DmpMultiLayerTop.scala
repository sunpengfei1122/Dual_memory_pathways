package dmpsnn

import chisel3._
import chisel3.util._

/** Multi-layer DMP-SNN network with sequential layer execution.
  *
  * Instantiates N DmpCore modules (one per layer) and N-1 SpikeVectorEncoder
  * modules (for inter-layer spike conversion). A top-level FSM sequences
  * layer execution: Layer0 → Encode → Layer1 → Encode → ... → LayerN-1 → Output.
  *
  * This matches the paper's edge-computing architecture where layers process
  * sequentially within each timestep.
  */
class DmpMultiLayerTop(netCfg: DmpNetworkConfig) extends Module {
  val io = IO(new Bundle {
    // AER input to first layer
    val aerIn = new Bundle {
      val valid = Input(Bool())
      val ready = Output(Bool())
      val event = Input(new AerEvent(netCfg.layers.head.spikeAddrBits))
      val last  = Input(Bool())
    }

    // AER output from last layer (spike events)
    val aerOut = new Bundle {
      val valid = Output(Bool())
      val ready = Input(Bool())
      val event = Output(new AerEvent(netCfg.layers.last.neuronAddrBits))
      val last  = Output(Bool())
    }

    // Mean membrane readout from last layer (for classification)
    val membraneOut = new Bundle {
      val valid = Output(Bool())
      val data  = Output(Vec(netCfg.outputNeurons, SInt(netCfg.layers.last.uBits.W)))
    }

    // Per-layer weight loading
    val weightLoad = new Bundle {
      val valid    = Input(Bool())
      val ready    = Output(Bool())
      val layerSel = Input(UInt(log2Ceil(math.max(netCfg.numLayers, 2)).W))
      val target   = Input(UInt(3.W))
      val addr     = Input(UInt(16.W))
      val data     = Input(UInt(netCfg.layers.map(c => math.max(c.fusedNeurons, c.memDim) * c.wBits).max.W))
    }

    // Status
    val busy        = Output(Bool())
    val timestep    = Output(UInt(32.W))
    val activeLayer = Output(UInt(log2Ceil(math.max(netCfg.numLayers, 2)).W))
  })

  // ===== Instantiate cores (one per layer) =====
  val cores = netCfg.layers.map(cfg => Module(new DmpCore(cfg)))

  // ===== Instantiate inter-layer spike encoders (N-1) =====
  val encoders = (0 until netCfg.numLayers - 1).map { i =>
    Module(new SpikeVectorEncoder(
      numNeurons = netCfg.layers(i).nNeurons,
      maxSpikes  = netCfg.layers(i + 1).maxSpikesPerStep
    ))
  }

  // ===== Shared spike buffer (sized for largest layer) =====
  val maxSpikes   = netCfg.maxMaxSpikes
  val maxAddrBits = netCfg.maxSpikeAddrBits
  val spikeBuffer = RegInit(VecInit(Seq.fill(maxSpikes)(0.U(maxAddrBits.W))))
  val spikeCount  = RegInit(0.U(log2Ceil(maxSpikes + 1).W))

  // ===== Timestep counter =====
  val timestepCounter = RegInit(0.U(32.W))

  // ===== Main FSM =====
  val sIdle :: sCollect :: sRunLayer :: sEncode :: sOutput :: Nil = Enum(5)
  val state    = RegInit(sIdle)
  val layerIdx = RegInit(0.U(log2Ceil(math.max(netCfg.numLayers, 2)).W))

  // ===== Output spike encoder (for AER output from last layer) =====
  val outSpikeIdx  = RegInit(0.U(log2Ceil(netCfg.outputNeurons + 1).W))
  val outSpikesReg = RegInit(VecInit(Seq.fill(netCfg.outputNeurons)(false.B)))

  // ===== Default signal assignments =====
  io.aerIn.ready      := state === sIdle || state === sCollect
  io.aerOut.valid      := false.B
  io.aerOut.event.addr := 0.U
  io.aerOut.last       := false.B
  io.membraneOut.valid := false.B
  io.membraneOut.data  := cores.last.io.membraneReadout
  io.busy              := state =/= sIdle
  io.timestep          := timestepCounter
  io.activeLayer       := layerIdx

  // ===== Weight load routing =====
  val wlReady = Wire(Vec(netCfg.numLayers, Bool()))
  for (i <- 0 until netCfg.numLayers) {
    cores(i).io.weightLoad.valid  := io.weightLoad.valid && (io.weightLoad.layerSel === i.U)
    cores(i).io.weightLoad.target := io.weightLoad.target
    cores(i).io.weightLoad.addr   := io.weightLoad.addr
    val coreDataWidth = math.max(netCfg.layers(i).fusedNeurons, netCfg.layers(i).memDim) * netCfg.layers(i).wBits
    cores(i).io.weightLoad.data   := io.weightLoad.data(coreDataWidth - 1, 0)
    wlReady(i) := cores(i).io.weightLoad.ready
  }
  io.weightLoad.ready := MuxLookup(io.weightLoad.layerSel, false.B)(
    (0 until netCfg.numLayers).map(i => i.U -> wlReady(i))
  )

  // ===== Default core inputs (all idle) =====
  for (i <- 0 until netCfg.numLayers) {
    val cfg_i = netCfg.layers(i)
    cores(i).io.inSpikes.valid := false.B
    cores(i).io.inSpikes.count := 0.U
    for (j <- 0 until cfg_i.maxSpikesPerStep) {
      cores(i).io.inSpikes.addrs(j) := 0.U
    }
  }

  // ===== Default encoder inputs (all idle) =====
  for (i <- 0 until netCfg.numLayers - 1) {
    encoders(i).io.start := false.B
    encoders(i).io.spikeVec := cores(i).io.outSpikes.vector
  }

  // ===== FSM Logic =====
  switch(state) {
    is(sIdle) {
      spikeCount := 0.U
      layerIdx := 0.U
      when(io.aerIn.valid) {
        spikeBuffer(0) := io.aerIn.event.addr
        spikeCount := 1.U
        state := Mux(io.aerIn.last, sRunLayer, sCollect)
      }
    }

    is(sCollect) {
      when(io.aerIn.valid) {
        when(spikeCount < maxSpikes.U) {
          spikeBuffer(spikeCount) := io.aerIn.event.addr
          spikeCount := spikeCount + 1.U
        }
        when(io.aerIn.last) {
          state := sRunLayer
        }
      }
    }

    is(sRunLayer) {
      // Drive the active core's input from the spike buffer
      for (i <- 0 until netCfg.numLayers) {
        when(layerIdx === i.U) {
          val cfg_i = netCfg.layers(i)
          cores(i).io.inSpikes.valid := true.B
          cores(i).io.inSpikes.count := spikeCount(log2Ceil(cfg_i.maxSpikesPerStep + 1) - 1, 0)
          for (j <- 0 until cfg_i.maxSpikesPerStep) {
            cores(i).io.inSpikes.addrs(j) := spikeBuffer(j)(cfg_i.spikeAddrBits - 1, 0)
          }
        }
      }

      // Wait for active core to produce output
      val coreDone = Wire(Bool())
      coreDone := false.B
      for (i <- 0 until netCfg.numLayers) {
        when(layerIdx === i.U) {
          coreDone := cores(i).io.outSpikes.valid
        }
      }

      when(coreDone) {
        when(layerIdx === (netCfg.numLayers - 1).U) {
          // Last layer done → output
          outSpikesReg := VecInit(cores.last.io.outSpikes.vector.map(_.asBool))
          outSpikeIdx := 0.U
          state := sOutput
        }.otherwise {
          // Start encoding for inter-layer transfer
          state := sEncode
        }
      }
    }

    is(sEncode) {
      // Start the encoder for current layer
      for (i <- 0 until netCfg.numLayers - 1) {
        when(layerIdx === i.U) {
          encoders(i).io.start := true.B
          encoders(i).io.spikeVec := cores(i).io.outSpikes.vector
        }
      }

      // Wait for encoder to complete
      val encDone = Wire(Bool())
      encDone := false.B
      for (i <- 0 until netCfg.numLayers - 1) {
        when(layerIdx === i.U) {
          encDone := encoders(i).io.done
        }
      }

      when(encDone) {
        // Transfer encoded spikes to the shared buffer for next layer
        for (i <- 0 until netCfg.numLayers - 1) {
          when(layerIdx === i.U) {
            spikeCount := encoders(i).io.count.pad(log2Ceil(maxSpikes + 1))
            for (j <- 0 until netCfg.layers(i + 1).maxSpikesPerStep) {
              spikeBuffer(j) := encoders(i).io.addrs(j).pad(maxAddrBits)
            }
          }
        }
        layerIdx := layerIdx + 1.U
        state := sRunLayer
      }
    }

    is(sOutput) {
      // Emit membrane readout
      io.membraneOut.valid := true.B

      // Emit AER output spikes (scan for fired neurons)
      val foundSpike = Wire(Bool())
      val nextIdx = Wire(UInt(log2Ceil(netCfg.outputNeurons + 1).W))
      foundSpike := false.B
      nextIdx := outSpikeIdx

      for (i <- netCfg.outputNeurons - 1 to 0 by -1) {
        when(i.U >= outSpikeIdx && outSpikesReg(i)) {
          foundSpike := true.B
          nextIdx := i.U
        }
      }

      when(foundSpike) {
        io.aerOut.valid := true.B
        io.aerOut.event.addr := nextIdx(netCfg.layers.last.neuronAddrBits - 1, 0)

        when(io.aerOut.ready) {
          outSpikesReg(nextIdx) := false.B
          outSpikeIdx := nextIdx + 1.U

          val anyRemaining = outSpikesReg.zipWithIndex.map { case (s, i) =>
            s && i.U > nextIdx
          }.reduce(_ || _)
          when(!anyRemaining) {
            io.aerOut.last := true.B
            timestepCounter := timestepCounter + 1.U
            state := sIdle
          }
        }
      }.otherwise {
        // No spikes — still emit a last marker
        io.aerOut.valid := true.B
        io.aerOut.event.addr := 0.U
        io.aerOut.last := true.B
        when(io.aerOut.ready) {
          timestepCounter := timestepCounter + 1.U
          state := sIdle
        }
      }
    }
  }
}
