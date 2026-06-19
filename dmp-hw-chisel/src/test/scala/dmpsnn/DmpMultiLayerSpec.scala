package dmpsnn

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class DmpMultiLayerSpec extends AnyFlatSpec with ChiselScalatestTester {
  val netCfg = DmpNetworkConfig(
    layers = Seq(
      DmpConfig(nNeurons = 8, memDim = 2, inputWidth = 16, maxSpikesPerStep = 4, fusedNeurons = 2),
      DmpConfig(nNeurons = 4, memDim = 2, inputWidth = 8, maxSpikesPerStep = 4, fusedNeurons = 2)
    ),
    maxSpikesInterLayer = 4
  )

  behavior of "DmpMultiLayerTop"

  it should "process spike input through both layers" in {
    test(new DmpMultiLayerTop(netCfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // Send AER input: 2 spike events at addresses 0 and 3
      dut.io.aerIn.valid.poke(true.B)
      dut.io.aerIn.event.addr.poke(0.U)
      dut.io.aerIn.last.poke(false.B)
      dut.clock.step(1)

      dut.io.aerIn.event.addr.poke(3.U)
      dut.io.aerIn.last.poke(true.B)
      dut.clock.step(1)

      dut.io.aerIn.valid.poke(false.B)
      dut.io.aerIn.last.poke(false.B)

      // Wait for processing to complete (both layers + encoding)
      dut.io.aerOut.ready.poke(true.B)
      var cycles = 0
      while (dut.io.busy.peekBoolean() && cycles < 1000) {
        dut.clock.step(1)
        cycles += 1
      }

      assert(!dut.io.busy.peekBoolean(), s"Multi-layer did not complete within $cycles cycles")
      println(s"Multi-layer processing completed in $cycles cycles")
      dut.io.timestep.expect(1.U)
    }
  }

  it should "handle zero-spike timestep through all layers" in {
    test(new DmpMultiLayerTop(netCfg)) { dut =>
      // Send empty timestep (valid + last immediately)
      dut.io.aerIn.valid.poke(true.B)
      dut.io.aerIn.event.addr.poke(0.U)
      dut.io.aerIn.last.poke(true.B)
      dut.clock.step(1)

      dut.io.aerIn.valid.poke(false.B)
      dut.io.aerIn.last.poke(false.B)
      dut.io.aerOut.ready.poke(true.B)

      var cycles = 0
      while (dut.io.busy.peekBoolean() && cycles < 1000) {
        dut.clock.step(1)
        cycles += 1
      }

      assert(!dut.io.busy.peekBoolean(), s"Zero-spike did not complete within $cycles cycles")
      dut.io.timestep.expect(1.U)
      println(s"Zero-spike multi-layer completed in $cycles cycles")
    }
  }

  it should "process multiple consecutive timesteps" in {
    test(new DmpMultiLayerTop(netCfg)) { dut =>
      dut.io.aerOut.ready.poke(true.B)

      for (t <- 0 until 3) {
        // Send one spike per timestep
        dut.io.aerIn.valid.poke(true.B)
        dut.io.aerIn.event.addr.poke(t.U)
        dut.io.aerIn.last.poke(true.B)
        dut.clock.step(1)

        dut.io.aerIn.valid.poke(false.B)
        dut.io.aerIn.last.poke(false.B)

        var cycles = 0
        while (dut.io.busy.peekBoolean() && cycles < 1000) {
          dut.clock.step(1)
          cycles += 1
        }

        assert(!dut.io.busy.peekBoolean(), s"Timestep $t did not complete within $cycles cycles")
      }

      dut.io.timestep.expect(3.U)
      println("Successfully processed 3 consecutive timesteps through 2 layers")
    }
  }

  it should "provide membrane readout from last layer" in {
    test(new DmpMultiLayerTop(netCfg)) { dut =>
      dut.io.aerOut.ready.poke(true.B)

      // Send spikes
      dut.io.aerIn.valid.poke(true.B)
      dut.io.aerIn.event.addr.poke(0.U)
      dut.io.aerIn.last.poke(true.B)
      dut.clock.step(1)

      dut.io.aerIn.valid.poke(false.B)
      dut.io.aerIn.last.poke(false.B)

      // Monitor for membrane readout validity during output phase
      var membraneValid = false
      var cycles = 0
      while (dut.io.busy.peekBoolean() && cycles < 1000) {
        if (dut.io.membraneOut.valid.peekBoolean()) {
          membraneValid = true
        }
        dut.clock.step(1)
        cycles += 1
      }

      assert(membraneValid, "Membrane readout should be valid during output")
      println(s"Membrane readout observed during output phase")
    }
  }
}
