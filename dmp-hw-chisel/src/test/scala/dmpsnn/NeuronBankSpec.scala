package dmpsnn

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class NeuronBankSpec extends AnyFlatSpec with ChiselScalatestTester {
  val cfg = DmpConfig(nNeurons = 8, memDim = 2, inputWidth = 16, maxSpikesPerStep = 4,
    fusedNeurons = 2, beta = 128, threshold = 100)  // β=0.5 for easy math

  behavior of "NeuronBank"

  it should "apply leak and accumulate inputs" in {
    test(new NeuronBank(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // Provide input currents: all neurons get I_spike=50, I_mem=0
      for (i <- 0 until cfg.nNeurons) {
        dut.io.iSpike(i).poke(50.S)
        dut.io.iMem(i).poke(0.S)
      }

      // Start neuron update
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      // Simulate neuron SRAM returning initial membrane = 0
      var cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 50) {
        for (i <- 0 until cfg.fusedNeurons) {
          dut.io.neuronSram.rData(i).poke(0.S)
        }
        dut.clock.step(1)
        cycles += 1
      }

      assert(dut.io.done.peekBoolean(), s"NeuronBank did not complete in $cycles cycles")

      // With u_old=0, β=0.5: leak=0, I_spike=50, I_mem=0 → u_new=50 < threshold=100
      // So no spikes should fire
      for (i <- 0 until cfg.nNeurons) {
        assert(!dut.io.spikesOut(i).peekBoolean(), s"Neuron $i should not fire")
      }
      println(s"NeuronBank completed in $cycles cycles, no spikes (correct)")
    }
  }

  it should "fire spikes when threshold is exceeded" in {
    test(new NeuronBank(cfg)) { dut =>
      // High input current that exceeds threshold
      for (i <- 0 until cfg.nNeurons) {
        dut.io.iSpike(i).poke(150.S)
        dut.io.iMem(i).poke(0.S)
      }

      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      var cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 50) {
        for (i <- 0 until cfg.fusedNeurons) {
          dut.io.neuronSram.rData(i).poke(0.S)
        }
        dut.clock.step(1)
        cycles += 1
      }

      assert(dut.io.done.peekBoolean())
      // u_new = 0*0.5 + 150 = 150 > 100 → all should fire
      for (i <- 0 until cfg.nNeurons) {
        assert(dut.io.spikesOut(i).peekBoolean(), s"Neuron $i should fire")
      }
      println(s"NeuronBank: all neurons fired (correct)")
    }
  }
}
