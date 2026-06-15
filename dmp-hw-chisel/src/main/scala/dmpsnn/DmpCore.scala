package dmpsnn

import chisel3._
import chisel3.util._

/** DMP-SNN Core: Top-level FSM orchestrating the four parallel datapaths.
  *
  * Execution flow per timestep:
  *   1. Receive spike addresses from input
  *   2. Launch in parallel (no data dependencies):
  *      - Path 1: SpikeIntegration (Wf · s)
  *      - Path 2: MemoryIntegration Path A (P · m[k-1])
  *      - Path 4: MemoryUpdate (Ā · m[k-1] + B̄ · x[k])
  *      - ScalarDrive (Wx · s + b → x[k])
  *   3. When ScalarDrive completes:
  *      - Path 3: MemoryIntegration Path B (v · x[k]) — folded into MemInteg
  *      - MemoryUpdate receives x[k]
  *   4. When Paths 1, 2+3 all complete:
  *      - NeuronBank: fused leak + accumulate + threshold + reset
  *   5. Output spikes
  */
class DmpCore(cfg: DmpConfig) extends Module {
  val io = IO(new Bundle {
    // Input spike interface
    val inSpikes = new Bundle {
      val valid = Input(Bool())
      val ready = Output(Bool())
      val count = Input(UInt(log2Ceil(cfg.maxSpikesPerStep + 1).W))
      val addrs = Input(Vec(cfg.maxSpikesPerStep, UInt(cfg.spikeAddrBits.W)))
    }

    // Output spikes
    val outSpikes = new Bundle {
      val valid  = Output(Bool())
      val vector = Output(Vec(cfg.nNeurons, Bool()))
    }

    // Status
    val busy = Output(Bool())
    val timestep = Output(UInt(32.W))
  })

  // ===== Instantiate submodules =====
  val spikeInteg = Module(new SpikeIntegration(cfg))
  val scalarDrive = Module(new ScalarDrive(cfg))
  val memUpdate = Module(new MemoryUpdate(cfg))
  val memInteg = Module(new MemoryIntegration(cfg))
  val neuronBank = Module(new NeuronBank(cfg))

  // ===== Weight SRAMs =====
  // Wf SRAM: inputWidth * neuronGroups rows, each row = fusedNeurons weights
  val wfSram = Module(new WideSram(
    depth = cfg.inputWidth * cfg.neuronGroups,
    wordWidth = cfg.wBits,
    vecWidth = cfg.fusedNeurons
  ))

  // Wx SRAM: inputWidth entries, wBits each
  val wxSram = Module(new Sram(depth = cfg.inputWidth, width = cfg.wBits))

  // P SRAM: neuronGroups * memDim rows, each row = fusedNeurons weights
  val pSram = Module(new WideSram(
    depth = cfg.neuronGroups * cfg.memDim,
    wordWidth = cfg.wBits,
    vecWidth = cfg.fusedNeurons
  ))

  // v SRAM: neuronGroups rows, each row = fusedNeurons weights
  val vSram = Module(new WideSram(
    depth = cfg.neuronGroups,
    wordWidth = cfg.wBits,
    vecWidth = cfg.fusedNeurons
  ))

  // Ā SRAM: memDim rows, each row = memDim weights (one full matrix row)
  val aBarSram = Module(new WideSram(
    depth = cfg.memDim,
    wordWidth = cfg.wBits,
    vecWidth = cfg.memDim
  ))

  // Neuron state SRAM: neuronGroups rows, each row = fusedNeurons membrane potentials
  val neuronSram = Module(new WideSram(
    depth = cfg.neuronGroups,
    wordWidth = cfg.uBits,
    vecWidth = cfg.fusedNeurons
  ))

  // ===== Spike address buffer =====
  val spikeAddrs = RegInit(VecInit(Seq.fill(cfg.maxSpikesPerStep)(0.U(cfg.spikeAddrBits.W))))
  val spikeCount = RegInit(0.U(log2Ceil(cfg.maxSpikesPerStep + 1).W))
  val timestepCounter = RegInit(0.U(32.W))

  // ===== Config registers (B̄ vector, bias) =====
  val bBarReg = RegInit(VecInit(Seq.fill(cfg.memDim)(0.S(cfg.wBits.W))))
  val biasReg = RegInit(0.S(cfg.mBits.W))

  // ===== Main FSM =====
  val sIdle :: sCompute :: sNeuronUpdate :: sOutput :: Nil = Enum(4)
  val state = RegInit(sIdle)

  io.busy := state =/= sIdle
  io.timestep := timestepCounter
  io.inSpikes.ready := state === sIdle
  io.outSpikes.valid := false.B
  io.outSpikes.vector := neuronBank.io.spikesOut

  // ===== Wire SRAMs to submodules =====

  // Wf SRAM ← SpikeIntegration
  wfSram.io.en := spikeInteg.io.wfSram.en
  wfSram.io.wen := false.B
  wfSram.io.addr := spikeInteg.io.wfSram.addr
  wfSram.io.wdata := VecInit(Seq.fill(cfg.fusedNeurons)(0.U(cfg.wBits.W)))
  spikeInteg.io.wfSram.rdata := VecInit(wfSram.io.rdata.map(_.asSInt))

  // Wx SRAM ← ScalarDrive
  wxSram.io.en := scalarDrive.io.wxSram.en
  wxSram.io.wen := false.B
  wxSram.io.addr := scalarDrive.io.wxSram.addr
  wxSram.io.wdata := 0.U
  scalarDrive.io.wxSram.rdata := wxSram.io.rdata.asSInt

  // P SRAM ← MemoryIntegration
  pSram.io.en := memInteg.io.pSram.en
  pSram.io.wen := false.B
  pSram.io.addr := memInteg.io.pSram.addr
  pSram.io.wdata := VecInit(Seq.fill(cfg.fusedNeurons)(0.U(cfg.wBits.W)))
  memInteg.io.pSram.rdata := VecInit(pSram.io.rdata.map(_.asSInt))

  // v SRAM ← MemoryIntegration
  vSram.io.en := memInteg.io.vSram.en
  vSram.io.wen := false.B
  vSram.io.addr := memInteg.io.vSram.addr
  vSram.io.wdata := VecInit(Seq.fill(cfg.fusedNeurons)(0.U(cfg.wBits.W)))
  memInteg.io.vSram.rdata := VecInit(vSram.io.rdata.map(_.asSInt))

  // Ā SRAM ← MemoryUpdate
  aBarSram.io.en := memUpdate.io.aBarSram.en
  aBarSram.io.wen := false.B
  aBarSram.io.addr := memUpdate.io.aBarSram.addr
  aBarSram.io.wdata := VecInit(Seq.fill(cfg.memDim)(0.U(cfg.wBits.W)))
  memUpdate.io.aBarSram.rdata := VecInit(aBarSram.io.rdata.map(_.asSInt))

  // Neuron SRAM ← NeuronBank (dual-port emulated with arbitration)
  neuronSram.io.en := neuronBank.io.neuronSram.rEn || neuronBank.io.neuronSram.wEn
  neuronSram.io.wen := neuronBank.io.neuronSram.wEn
  neuronSram.io.addr := Mux(neuronBank.io.neuronSram.wEn,
    neuronBank.io.neuronSram.wAddr,
    neuronBank.io.neuronSram.rAddr)
  neuronSram.io.wdata := VecInit(neuronBank.io.neuronSram.wData.map(_.asUInt))
  neuronBank.io.neuronSram.rData := VecInit(neuronSram.io.rdata.map(_.asSInt))

  // ===== Connect submodule inputs =====

  // Spike address mux: both SpikeInteg and ScalarDrive read from the same buffer
  spikeInteg.io.spikeCount := spikeCount
  spikeInteg.io.spikeAddr := spikeAddrs(spikeInteg.io.spikeIdx)
  scalarDrive.io.spikeCount := spikeCount
  scalarDrive.io.spikeAddr := spikeAddrs(scalarDrive.io.spikeIdx)
  scalarDrive.io.bias := biasReg

  // Memory modules
  memUpdate.io.xIn := scalarDrive.io.xOut
  memUpdate.io.xInValid := scalarDrive.io.xOutValid
  memUpdate.io.bBar := bBarReg
  memInteg.io.mPrev := memUpdate.io.mStatePrev
  memInteg.io.xIn := scalarDrive.io.xOut
  memInteg.io.xInValid := scalarDrive.io.xOutValid

  // Neuron bank inputs
  neuronBank.io.iSpike := spikeInteg.io.result
  neuronBank.io.iMem := memInteg.io.result

  // ===== Start/done control signals =====
  val computeStart = RegInit(false.B)
  val neuronStart  = RegInit(false.B)

  spikeInteg.io.start := computeStart
  scalarDrive.io.start := computeStart
  memUpdate.io.start := computeStart
  memInteg.io.start := computeStart
  neuronBank.io.start := neuronStart

  // All parallel paths done
  val allPathsDone = spikeInteg.io.done && memInteg.io.done

  switch(state) {
    is(sIdle) {
      computeStart := false.B
      neuronStart := false.B
      when(io.inSpikes.valid) {
        // Latch spike addresses
        spikeCount := io.inSpikes.count
        for (i <- 0 until cfg.maxSpikesPerStep) {
          spikeAddrs(i) := io.inSpikes.addrs(i)
        }
        state := sCompute
        computeStart := true.B
      }
    }

    is(sCompute) {
      computeStart := false.B
      // Wait for all parallel integration paths to complete
      when(allPathsDone) {
        neuronStart := true.B
        state := sNeuronUpdate
      }
    }

    is(sNeuronUpdate) {
      neuronStart := false.B
      when(neuronBank.io.done) {
        state := sOutput
      }
    }

    is(sOutput) {
      io.outSpikes.valid := true.B
      timestepCounter := timestepCounter + 1.U
      state := sIdle
    }
  }
}
