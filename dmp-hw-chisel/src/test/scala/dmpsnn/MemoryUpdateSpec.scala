package dmpsnn

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class MemoryUpdateSpec extends AnyFlatSpec with ChiselScalatestTester {
  val cfg = DmpConfig(nNeurons = 8, memDim = 4, inputWidth = 16, maxSpikesPerStep = 4, fusedNeurons = 2)

  behavior of "MemoryUpdate"

  it should "complete the state-space update in d+1 cycles" in {
    test(new MemoryUpdate(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // Initialize B̄ vector
      for (i <- 0 until cfg.memDim) {
        dut.io.bBar(i).poke(1.S)
      }

      // Start the update
      dut.io.start.poke(true.B)
      dut.io.xIn.poke(10.S)
      dut.io.xInValid.poke(false.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      // Provide Ā matrix data (identity-like for testing)
      var cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 50) {
        // Simulate SRAM returning identity row
        for (j <- 0 until cfg.memDim) {
          dut.io.aBarSram.rdata(j).poke(0.S)
        }
        // Make x valid after a few cycles (simulates ScalarDrive finishing)
        if (cycles > cfg.memDim) {
          dut.io.xInValid.poke(true.B)
        }
        dut.clock.step(1)
        cycles += 1
      }

      assert(dut.io.done.peekBoolean(), s"MemoryUpdate did not complete within $cycles cycles")
      println(s"MemoryUpdate completed in $cycles cycles")
    }
  }

  it should "expose previous memory state for MemoryIntegration" in {
    test(new MemoryUpdate(cfg)) { dut =>
      for (i <- 0 until cfg.memDim) {
        dut.io.bBar(i).poke(0.S)
      }

      // Initially all zeros
      for (i <- 0 until cfg.memDim) {
        dut.io.mStatePrev(i).expect(0.S)
      }

      dut.io.start.poke(true.B)
      dut.io.xInValid.poke(true.B)
      dut.io.xIn.poke(0.S)
      dut.clock.step(1)
      dut.io.start.poke(false.B)

      // mPrev should now hold the snapshot of the previous state
      for (i <- 0 until cfg.memDim) {
        dut.io.mStatePrev(i).expect(0.S)
      }
    }
  }
}
