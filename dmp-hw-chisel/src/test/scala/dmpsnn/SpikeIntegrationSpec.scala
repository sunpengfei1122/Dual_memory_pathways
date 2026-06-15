package dmpsnn

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SpikeIntegrationSpec extends AnyFlatSpec with ChiselScalatestTester {
  val cfg = DmpConfig(nNeurons = 8, memDim = 2, inputWidth = 16, maxSpikesPerStep = 4, fusedNeurons = 2)

  behavior of "SpikeIntegration"

  it should "accumulate weights for active spike addresses" in {
    test(new SpikeIntegration(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // Setup: 2 spikes at addresses 0 and 1
      dut.io.start.poke(true.B)
      dut.io.spikeCount.poke(2.U)
      dut.io.spikeAddr.poke(0.U)

      // Provide weight data via SRAM interface (simulated)
      // For spike addr 0: weights for neuron group 0 = [1, 2], group 1 = [3, 4], etc.
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      // Let it run through the FSM
      var cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 100) {
        // Simulate SRAM response (weights matching the spike address)
        for (i <- 0 until cfg.fusedNeurons) {
          dut.io.wfSram.rdata(i).poke(((i + 1) * 1).S)
        }
        dut.clock.step(1)
        cycles += 1
      }

      assert(dut.io.done.peekBoolean(), s"SpikeIntegration did not complete within $cycles cycles")
      println(s"SpikeIntegration completed in $cycles cycles")
    }
  }

  it should "produce zero output with zero spikes" in {
    test(new SpikeIntegration(cfg)) { dut =>
      dut.io.start.poke(true.B)
      dut.io.spikeCount.poke(0.U)
      dut.io.spikeAddr.poke(0.U)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      dut.clock.step(1)

      assert(dut.io.done.peekBoolean())
      for (i <- 0 until cfg.nNeurons) {
        dut.io.result(i).expect(0.S)
      }
    }
  }
}
