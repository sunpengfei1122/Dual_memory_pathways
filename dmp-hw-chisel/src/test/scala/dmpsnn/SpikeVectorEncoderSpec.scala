package dmpsnn

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SpikeVectorEncoderSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "SpikeVectorEncoder"

  it should "produce zero count with no spikes" in {
    test(new SpikeVectorEncoder(numNeurons = 8, maxSpikes = 4)) { dut =>
      // All zeros input
      for (i <- 0 until 8) {
        dut.io.spikeVec(i).poke(false.B)
      }
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      var cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 20) {
        dut.clock.step(1)
        cycles += 1
      }

      assert(dut.io.done.peekBoolean(), "Should complete quickly with no spikes")
      dut.io.count.expect(0.U)
      println(s"Zero-spike encoding completed in $cycles cycles")
    }
  }

  it should "encode sparse spike positions correctly" in {
    test(new SpikeVectorEncoder(numNeurons = 8, maxSpikes = 4)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // Spikes at indices 1, 4, 6
      for (i <- 0 until 8) dut.io.spikeVec(i).poke(false.B)
      dut.io.spikeVec(1).poke(true.B)
      dut.io.spikeVec(4).poke(true.B)
      dut.io.spikeVec(6).poke(true.B)

      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      var cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 20) {
        dut.clock.step(1)
        cycles += 1
      }

      assert(dut.io.done.peekBoolean())
      dut.io.count.expect(3.U)
      dut.io.addrs(0).expect(1.U)
      dut.io.addrs(1).expect(4.U)
      dut.io.addrs(2).expect(6.U)
      println(s"Encoded 3 spikes in $cycles cycles: [1, 4, 6]")
    }
  }

  it should "saturate at maxSpikes when more neurons fire" in {
    test(new SpikeVectorEncoder(numNeurons = 8, maxSpikes = 2)) { dut =>
      // All 8 neurons fire but maxSpikes = 2
      for (i <- 0 until 8) dut.io.spikeVec(i).poke(true.B)

      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      var cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 20) {
        dut.clock.step(1)
        cycles += 1
      }

      assert(dut.io.done.peekBoolean())
      dut.io.count.expect(2.U)
      // Should capture the lowest 2 indices (priority encoder)
      dut.io.addrs(0).expect(0.U)
      dut.io.addrs(1).expect(1.U)
      println(s"Saturated at maxSpikes=2 in $cycles cycles")
    }
  }
}
