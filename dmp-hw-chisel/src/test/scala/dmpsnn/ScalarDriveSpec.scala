package dmpsnn

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class ScalarDriveSpec extends AnyFlatSpec with ChiselScalatestTester {
  val cfg = DmpConfig(nNeurons = 8, memDim = 2, inputWidth = 16, maxSpikesPerStep = 4, fusedNeurons = 2)

  behavior of "ScalarDrive"

  it should "compute scalar drive with 2 spikes" in {
    test(new ScalarDrive(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // Spike addresses [3, 7] with weights 10 and 20
      val spikeAddrs = Seq(3, 7)
      val weights = Map(3 -> 10, 7 -> 20)
      val bias = 5

      dut.io.bias.poke(bias.S)
      dut.io.spikeCount.poke(2.U)
      dut.io.spikeAddr.poke(spikeAddrs(0).U)
      dut.io.wxSram.rdata.poke(0.S)

      // Start — on this edge, ScalarDrive transitions to sProcess
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      // Model the 1-cycle SRAM latency by tracking the address issued AFTER all
      // combinational updates settle (i.e., after spikeAddr is poked for this cycle).
      // That address's weight becomes rdata on the NEXT cycle.
      var issuedAddr = -1
      var cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 50) {
        // Provide rdata for the address issued last cycle
        if (issuedAddr >= 0) {
          dut.io.wxSram.rdata.poke(weights.getOrElse(issuedAddr, 0).S)
        }

        // Update spikeAddr based on current spikeIdx (models DmpCore's mux)
        val idx = dut.io.spikeIdx.peekInt().toInt
        if (idx < spikeAddrs.length) {
          dut.io.spikeAddr.poke(spikeAddrs(idx).U)
        }

        // Now peek the address after spikeAddr poke settles combinationally
        issuedAddr = dut.io.wxSram.addr.peekInt().toInt

        dut.clock.step(1)
        cycles += 1
      }

      assert(dut.io.done.peekBoolean(), s"ScalarDrive did not complete within $cycles cycles")
      println(s"ScalarDrive completed in $cycles cycles")

      // Pipeline drain in sDone: the module accumulates the last rdata
      dut.io.wxSram.rdata.poke(weights.getOrElse(issuedAddr, 0).S)
      dut.clock.step(1)

      // Output = sum(weights for all spike addrs) + bias = 10 + 20 + 5 = 35
      val expected = weights(3) + weights(7) + bias
      val xOut = dut.io.xOut.peekInt().toInt
      println(s"xOut = $xOut (expected $expected)")
      assert(xOut == expected, s"Expected $expected, got $xOut")
    }
  }

  it should "handle zero spikes" in {
    test(new ScalarDrive(cfg)) { dut =>
      dut.io.bias.poke(7.S)
      dut.io.spikeCount.poke(0.U)
      dut.io.spikeAddr.poke(0.U)

      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      // Should be done immediately
      assert(dut.io.done.peekBoolean(), "Should be done immediately with 0 spikes")
      // Output should be just the bias
      dut.io.xOut.expect(7.S)
      println(s"Zero spikes: xOut = ${dut.io.xOut.peekInt()} (expected 7)")
    }
  }

  it should "restart correctly from sDone state" in {
    test(new ScalarDrive(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.io.bias.poke(0.S)
      dut.io.wxSram.rdata.poke(5.S)

      // First run: 1 spike
      dut.io.spikeCount.poke(1.U)
      dut.io.spikeAddr.poke(0.U)
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      var cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 50) {
        dut.clock.step(1)
        cycles += 1
      }
      assert(dut.io.done.peekBoolean(), "First run should complete")
      println(s"First run done in $cycles cycles")

      // Wait one cycle (pipeline drain)
      dut.clock.step(1)

      // Second run: try to restart with a 1-cycle start pulse while in sDone
      dut.io.spikeCount.poke(1.U)
      dut.io.spikeAddr.poke(1.U)
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      // Check if it restarts properly
      cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 50) {
        dut.clock.step(1)
        cycles += 1
      }

      if (dut.io.done.peekBoolean()) {
        println(s"Second run completed in $cycles cycles - restart works")
      } else {
        println(s"BUG: Second run did NOT complete within $cycles cycles - stuck after restart!")
        assert(false, "ScalarDrive failed to restart from sDone")
      }
    }
  }
}
