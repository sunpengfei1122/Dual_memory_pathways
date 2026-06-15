package dmpsnn

import chisel3._
import chisel3.util._

/** Path 1: Sparse spike integration using input-stationary dataflow.
  *
  * For each active spike address `a`, reads the column Wf[:, a] from weight SRAM
  * and accumulates it into the neuron partial-sum buffer. Processes `fusedNeurons`
  * neurons per cycle (column-wise streaming).
  *
  * Input-stationary means: one spike address is held stationary while all neuron
  * groups are streamed through — minimizes weight-SRAM access for sparse inputs.
  */
class SpikeIntegration(cfg: DmpConfig) extends Module {
  val io = IO(new Bundle {
    val start     = Input(Bool())
    val done      = Output(Bool())

    // Spike input: address stream
    val spikeCount = Input(UInt(log2Ceil(cfg.maxSpikesPerStep + 1).W))
    val spikeAddr  = Input(UInt(cfg.spikeAddrBits.W))
    val spikeIdx   = Output(UInt(log2Ceil(cfg.maxSpikesPerStep + 1).W))

    // Weight SRAM interface (reads fusedNeurons weights per cycle)
    // Addressed as: [spikeAddr * neuronGroups + neuronGroup]
    val wfSram = new Bundle {
      val en   = Output(Bool())
      val addr = Output(UInt(log2Ceil(cfg.inputWidth * cfg.neuronGroups).W))
      val rdata = Input(Vec(cfg.fusedNeurons, SInt(cfg.wBits.W)))
    }

    // Output: partial sum for each neuron (accumulated across all spikes)
    val result    = Output(Vec(cfg.nNeurons, SInt(cfg.accBits.W)))
    val resultValid = Output(Bool())
  })

  // Accumulator registers for all neurons
  val acc = RegInit(VecInit(Seq.fill(cfg.nNeurons)(0.S(cfg.accBits.W))))

  // FSM
  val sIdle :: sProcess :: sDone :: Nil = Enum(3)
  val state = RegInit(sIdle)

  val spikeCounter = RegInit(0.U(log2Ceil(cfg.maxSpikesPerStep + 1).W))
  val neuronGroupCounter = RegInit(0.U(log2Ceil(cfg.neuronGroups + 1).W))

  // Pipeline: SRAM has 1-cycle read latency
  val neuronGroupDelayed = RegNext(neuronGroupCounter)
  val validPipe = RegInit(false.B)

  io.done := state === sDone
  io.spikeIdx := spikeCounter
  io.wfSram.en := false.B
  io.wfSram.addr := 0.U
  io.result := acc
  io.resultValid := state === sDone

  switch(state) {
    is(sIdle) {
      when(io.start) {
        // Clear accumulators
        for (i <- 0 until cfg.nNeurons) {
          acc(i) := 0.S
        }
        spikeCounter := 0.U
        neuronGroupCounter := 0.U
        validPipe := false.B
        state := Mux(io.spikeCount === 0.U, sDone, sProcess)
      }
    }

    is(sProcess) {
      // Issue SRAM read for current spike address, current neuron group
      io.wfSram.en := true.B
      io.wfSram.addr := io.spikeAddr * cfg.neuronGroups.U + neuronGroupCounter

      // Accumulate results from previous cycle's read
      when(validPipe) {
        for (i <- 0 until cfg.fusedNeurons) {
          val neuronIdx = neuronGroupDelayed * cfg.fusedNeurons.U + i.U
          acc(neuronIdx) := acc(neuronIdx) + io.wfSram.rdata(i).pad(cfg.accBits)
        }
      }

      validPipe := true.B

      // Advance counters
      when(neuronGroupCounter === (cfg.neuronGroups - 1).U) {
        neuronGroupCounter := 0.U
        when(spikeCounter === io.spikeCount - 1.U) {
          // Last spike, last group — but still need to accumulate the final read
          state := sDone
        }.otherwise {
          spikeCounter := spikeCounter + 1.U
        }
      }.otherwise {
        neuronGroupCounter := neuronGroupCounter + 1.U
      }
    }

    is(sDone) {
      // Accumulate the last pipelined read
      when(validPipe) {
        for (i <- 0 until cfg.fusedNeurons) {
          val neuronIdx = neuronGroupDelayed * cfg.fusedNeurons.U + i.U
          acc(neuronIdx) := acc(neuronIdx) + io.wfSram.rdata(i).pad(cfg.accBits)
        }
        validPipe := false.B
      }
      when(io.start) {
        state := sIdle
      }
    }
  }
}
