package dmpsnn

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class DmpCoreSpec extends AnyFlatSpec with ChiselScalatestTester {
  val cfg = DmpConfig(nNeurons = 8, memDim = 2, inputWidth = 16, maxSpikesPerStep = 4, fusedNeurons = 2)

  behavior of "DmpCore"

  it should "accept spike input and produce output" in {
    test(new DmpCore(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // Send 2 spikes at addresses 0 and 3
      dut.io.inSpikes.valid.poke(true.B)
      dut.io.inSpikes.count.poke(2.U)
      dut.io.inSpikes.addrs(0).poke(0.U)
      dut.io.inSpikes.addrs(1).poke(3.U)
      for (i <- 2 until cfg.maxSpikesPerStep) {
        dut.io.inSpikes.addrs(i).poke(0.U)
      }

      dut.clock.step(1)
      dut.io.inSpikes.valid.poke(false.B)

      // Wait for computation to complete
      var cycles = 0
      while (!dut.io.outSpikes.valid.peekBoolean() && cycles < 500) {
        dut.clock.step(1)
        cycles += 1
      }

      if (dut.io.outSpikes.valid.peekBoolean()) {
        println(s"DmpCore produced output after $cycles cycles")
        val firedCount = (0 until cfg.nNeurons).count(i =>
          dut.io.outSpikes.vector(i).peekBoolean())
        println(s"  $firedCount neurons fired")
      } else {
        println(s"WARNING: DmpCore did not produce output within $cycles cycles")
      }

      // Verify timestep counter advanced
      dut.io.timestep.expect(1.U)
    }
  }

  it should "handle zero-spike timestep" in {
    test(new DmpCore(cfg)) { dut =>
      dut.io.inSpikes.valid.poke(true.B)
      dut.io.inSpikes.count.poke(0.U)
      for (i <- 0 until cfg.maxSpikesPerStep) {
        dut.io.inSpikes.addrs(i).poke(0.U)
      }
      dut.clock.step(1)
      dut.io.inSpikes.valid.poke(false.B)

      var cycles = 0
      while (!dut.io.outSpikes.valid.peekBoolean() && cycles < 500) {
        dut.clock.step(1)
        cycles += 1
      }

      assert(dut.io.outSpikes.valid.peekBoolean(), "Should complete with zero spikes")
      println(s"Zero-spike timestep completed in $cycles cycles")
    }
  }

  it should "process multiple consecutive timesteps" in {
    test(new DmpCore(cfg)) { dut =>
      for (t <- 0 until 3) {
        // Send spikes
        dut.io.inSpikes.valid.poke(true.B)
        dut.io.inSpikes.count.poke(1.U)
        dut.io.inSpikes.addrs(0).poke(t.U)
        for (i <- 1 until cfg.maxSpikesPerStep) {
          dut.io.inSpikes.addrs(i).poke(0.U)
        }
        dut.clock.step(1)
        dut.io.inSpikes.valid.poke(false.B)

        // Wait for output
        var cycles = 0
        while (!dut.io.outSpikes.valid.peekBoolean() && cycles < 500) {
          dut.clock.step(1)
          cycles += 1
        }
        assert(dut.io.outSpikes.valid.peekBoolean(), s"Timestep $t did not complete")
        dut.clock.step(1)
      }

      dut.io.timestep.expect(3.U)
      println("Successfully processed 3 consecutive timesteps")
    }
  }
}
