package dmpsnn

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** Bit-accurate golden model matching the hardware's fixed-point arithmetic. */
object GoldenModel {
  def saturate(value: Long, bits: Int): Int = {
    val max = (1L << (bits - 1)) - 1
    val min = -(1L << (bits - 1))
    if (value > max) max.toInt
    else if (value < min) min.toInt
    else value.toInt
  }

  def relu(value: Long, bits: Int): Int = {
    if (value <= 0) 0
    else saturate(value, bits)
  }

  def leak(u: Int, beta: Int, shift: Int): Int = {
    // Matches hardware: (u.toInt * beta) >> shift with arithmetic right shift
    val product = u.toLong * beta.toLong
    (product >> shift).toInt
  }

  def spikeIntegration(
    spikeAddrs: Seq[Int],
    wf: Array[Array[Int]], // wf[neuronIdx][inputAddr]
    nNeurons: Int,
    accBits: Int
  ): Array[Int] = {
    val acc = Array.fill(nNeurons)(0L)
    for (addr <- spikeAddrs) {
      for (i <- 0 until nNeurons) {
        acc(i) += wf(i)(addr).toLong
      }
    }
    acc.map(v => saturate(v, accBits))
  }

  def scalarDrive(
    spikeAddrs: Seq[Int],
    wx: Array[Int], // wx[inputAddr]
    bias: Int,
    accBits: Int,
    mBits: Int
  ): Int = {
    var acc = 0L
    for (addr <- spikeAddrs) {
      acc += wx(addr).toLong
    }
    val xSum = acc + bias.toLong
    relu(xSum, mBits)
  }

  def memoryUpdate(
    mPrev: Array[Int],
    aBar: Array[Array[Int]], // aBar[row][col], d×d
    bBar: Array[Int],        // bBar[i], d×1
    x: Int,
    wBits: Int,
    mBits: Int,
    internalAccBits: Int
  ): Array[Int] = {
    val d = mPrev.length
    val mNew = Array.fill(d)(0L)
    // Ā · m[k-1]
    for (row <- 0 until d) {
      var dot = 0L
      for (col <- 0 until d) {
        dot += aBar(row)(col).toLong * mPrev(col).toLong
      }
      mNew(row) = dot
    }
    // + B̄ · x[k]
    for (i <- 0 until d) {
      val bx = bBar(i).toLong * x.toLong
      mNew(i) += bx
    }
    mNew.map(v => saturate(v, mBits))
  }

  def memoryIntegration(
    mPrev: Array[Int],
    p: Array[Array[Int]], // p[neuronIdx][memDim]
    v: Array[Int],        // v[neuronIdx]
    x: Int,
    nNeurons: Int,
    accBits: Int,
    internalAccBits: Int
  ): Array[Int] = {
    val acc = Array.fill(nNeurons)(0L)
    val d = mPrev.length
    // P · m[k-1]
    for (i <- 0 until nNeurons) {
      for (j <- 0 until d) {
        acc(i) += p(i)(j).toLong * mPrev(j).toLong
      }
    }
    // + v · x[k]
    for (i <- 0 until nNeurons) {
      acc(i) += v(i).toLong * x.toLong
    }
    acc.map(v => saturate(v, accBits))
  }

  def neuronUpdate(
    uOld: Array[Int],
    iSpike: Array[Int],
    iMem: Array[Int],
    beta: Int,
    betaShift: Int,
    threshold: Int,
    uBits: Int,
    accBits: Int
  ): (Array[Int], Array[Boolean]) = {
    val n = uOld.length
    val uNew = Array.fill(n)(0)
    val spikes = Array.fill(n)(false)
    for (i <- 0 until n) {
      val leaked = leak(uOld(i), beta, betaShift)
      // Saturate integration currents from accBits to uBits
      val iSpikeW = saturate(iSpike(i).toLong, uBits)
      val iMemW = saturate(iMem(i).toLong, uBits)
      val uRaw = leaked.toLong + iSpikeW.toLong + iMemW.toLong
      val fired = uRaw >= threshold
      val uReset = if (fired) uRaw - threshold else uRaw
      uNew(i) = saturate(uReset, uBits)
      spikes(i) = fired
    }
    (uNew, spikes)
  }

  case class LayerState(
    membrane: Array[Int],
    memory: Array[Int]
  )

  def timestep(
    state: LayerState,
    spikeAddrs: Seq[Int],
    wf: Array[Array[Int]],
    wx: Array[Int],
    p: Array[Array[Int]],
    v: Array[Int],
    aBar: Array[Array[Int]],
    bBar: Array[Int],
    bias: Int,
    cfg: DmpConfig
  ): LayerState = {
    val iSpike = spikeIntegration(spikeAddrs, wf, cfg.nNeurons, cfg.accBits)
    val x = scalarDrive(spikeAddrs, wx, bias, cfg.accBits, cfg.mBits)
    val mNew = memoryUpdate(state.memory, aBar, bBar, x, cfg.wBits, cfg.mBits, cfg.internalAccBits)
    val iMem = memoryIntegration(state.memory, p, v, x, cfg.nNeurons, cfg.accBits, cfg.internalAccBits)
    val (uNew, spikes) = neuronUpdate(
      state.membrane, iSpike, iMem,
      cfg.beta, BETA_SHIFT, cfg.threshold, cfg.uBits, cfg.accBits
    )
    LayerState(uNew, mNew)
  }
}

/** Test utilities for weight loading and result extraction. */
object TestUtils {
  def packWeights(weights: Seq[Int], wBits: Int): BigInt = {
    var packed = BigInt(0)
    for (i <- weights.indices) {
      val w = weights(i) & ((1 << wBits) - 1)
      packed = packed | (BigInt(w) << (i * wBits))
    }
    packed
  }

  def loadAllWeights(
    dut: DmpCore,
    cfg: DmpConfig,
    wf: Array[Array[Int]],   // wf[neuron][input]
    wx: Array[Int],          // wx[input]
    p: Array[Array[Int]],    // p[neuron][memDim]
    v: Array[Int],           // v[neuron]
    aBar: Array[Array[Int]], // aBar[row][col]
    bBar: Array[Int],        // bBar[dim]
    bias: Int
  ): Unit = {
    val neuronGroups = cfg.nNeurons / cfg.fusedNeurons

    // Target 0: Wf — address = inputAddr * neuronGroups + group
    for (inputAddr <- 0 until cfg.inputWidth) {
      for (group <- 0 until neuronGroups) {
        val weights = (0 until cfg.fusedNeurons).map { f =>
          val neuronIdx = group * cfg.fusedNeurons + f
          wf(neuronIdx)(inputAddr)
        }
        val addr = inputAddr * neuronGroups + group
        dut.io.weightLoad.valid.poke(true.B)
        dut.io.weightLoad.target.poke(0.U)
        dut.io.weightLoad.addr.poke(addr.U)
        dut.io.weightLoad.data.poke(packWeights(weights, cfg.wBits).U)
        dut.clock.step(1)
      }
    }

    // Target 1: Wx — address = inputAddr
    for (inputAddr <- 0 until cfg.inputWidth) {
      dut.io.weightLoad.valid.poke(true.B)
      dut.io.weightLoad.target.poke(1.U)
      dut.io.weightLoad.addr.poke(inputAddr.U)
      dut.io.weightLoad.data.poke((wx(inputAddr) & ((1 << cfg.wBits) - 1)).U)
      dut.clock.step(1)
    }

    // Target 2: P — address = group * memDim + dim
    for (group <- 0 until neuronGroups) {
      for (dim <- 0 until cfg.memDim) {
        val weights = (0 until cfg.fusedNeurons).map { f =>
          val neuronIdx = group * cfg.fusedNeurons + f
          p(neuronIdx)(dim)
        }
        val addr = group * cfg.memDim + dim
        dut.io.weightLoad.valid.poke(true.B)
        dut.io.weightLoad.target.poke(2.U)
        dut.io.weightLoad.addr.poke(addr.U)
        dut.io.weightLoad.data.poke(packWeights(weights, cfg.wBits).U)
        dut.clock.step(1)
      }
    }

    // Target 3: v — address = group
    for (group <- 0 until neuronGroups) {
      val weights = (0 until cfg.fusedNeurons).map { f =>
        val neuronIdx = group * cfg.fusedNeurons + f
        v(neuronIdx)
      }
      dut.io.weightLoad.valid.poke(true.B)
      dut.io.weightLoad.target.poke(3.U)
      dut.io.weightLoad.addr.poke(group.U)
      dut.io.weightLoad.data.poke(packWeights(weights, cfg.wBits).U)
      dut.clock.step(1)
    }

    // Target 4: Ā — address = row, data = full row (memDim elements)
    for (row <- 0 until cfg.memDim) {
      val weights = (0 until cfg.memDim).map(col => aBar(row)(col))
      dut.io.weightLoad.valid.poke(true.B)
      dut.io.weightLoad.target.poke(4.U)
      dut.io.weightLoad.addr.poke(row.U)
      dut.io.weightLoad.data.poke(packWeights(weights, cfg.wBits).U)
      dut.clock.step(1)
    }

    // Target 5: B̄ — one element at a time
    for (i <- 0 until cfg.memDim) {
      dut.io.weightLoad.valid.poke(true.B)
      dut.io.weightLoad.target.poke(5.U)
      dut.io.weightLoad.addr.poke(i.U)
      dut.io.weightLoad.data.poke((bBar(i) & ((1 << cfg.wBits) - 1)).U)
      dut.clock.step(1)
    }

    // Target 6: bias
    dut.io.weightLoad.valid.poke(true.B)
    dut.io.weightLoad.target.poke(6.U)
    dut.io.weightLoad.addr.poke(0.U)
    dut.io.weightLoad.data.poke((bias & ((1 << cfg.mBits) - 1)).U)
    dut.clock.step(1)

    dut.io.weightLoad.valid.poke(false.B)
    dut.clock.step(1)
  }

  def runTimestep(dut: DmpCore, cfg: DmpConfig, spikeAddrs: Seq[Int], maxCycles: Int = 1000): Unit = {
    // Ensure we're in idle before sending spikes
    assert(!dut.io.busy.peekBoolean(), "DmpCore is not idle at start of runTimestep")

    dut.io.inSpikes.valid.poke(true.B)
    dut.io.inSpikes.count.poke(spikeAddrs.length.U)
    for (i <- 0 until cfg.maxSpikesPerStep) {
      val addr = if (i < spikeAddrs.length) spikeAddrs(i) else 0
      dut.io.inSpikes.addrs(i).poke(addr.U)
    }
    dut.clock.step(1)
    dut.io.inSpikes.valid.poke(false.B)

    var cycles = 0
    while (!dut.io.outSpikes.valid.peekBoolean() && cycles < maxCycles) {
      dut.clock.step(1)
      cycles += 1
    }
    assert(dut.io.outSpikes.valid.peekBoolean(), s"DmpCore did not complete within $cycles cycles")
  }
}

class DmpGoldenModelSpec extends AnyFlatSpec with ChiselScalatestTester {
  val cfg = DmpConfig(
    nNeurons = 8,
    memDim = 2,
    inputWidth = 8,
    maxSpikesPerStep = 4,
    wBits = 8,
    uBits = 16,
    mBits = 16,
    accBits = 24,
    beta = 230,
    threshold = 512,
    fusedNeurons = 2
  )

  // Deterministic weight generation
  def genWf(nNeurons: Int, inputWidth: Int): Array[Array[Int]] =
    Array.tabulate(nNeurons, inputWidth)((i, j) => ((i + j) % 5) - 2)

  def genWx(inputWidth: Int): Array[Int] =
    Array.tabulate(inputWidth)(j => (j % 3) + 1)

  def genP(nNeurons: Int, memDim: Int): Array[Array[Int]] =
    Array.tabulate(nNeurons, memDim)((i, j) => ((i * 2 + j) % 5) - 2)

  def genV(nNeurons: Int): Array[Int] =
    Array.tabulate(nNeurons)(i => (i % 4) - 1)

  def genABar(memDim: Int): Array[Array[Int]] = {
    // Simple diagonal-dominant matrix (scaled for 8-bit fixed point)
    Array.tabulate(memDim, memDim)((i, j) => if (i == j) 120 else -10)
  }

  def genBBar(memDim: Int): Array[Int] =
    Array.fill(memDim)(1)

  behavior of "DmpCore golden model"

  it should "produce correct membrane potentials with known weights" in {
    test(new DmpCore(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      val wf = genWf(cfg.nNeurons, cfg.inputWidth)
      val wx = genWx(cfg.inputWidth)
      val p = genP(cfg.nNeurons, cfg.memDim)
      val v = genV(cfg.nNeurons)
      val aBar = genABar(cfg.memDim)
      val bBar = genBBar(cfg.memDim)
      val bias = 3

      TestUtils.loadAllWeights(dut, cfg, wf, wx, p, v, aBar, bBar, bias)

      val spikeAddrs = Seq(1, 3)
      val initState = GoldenModel.LayerState(
        membrane = Array.fill(cfg.nNeurons)(0),
        memory = Array.fill(cfg.memDim)(0)
      )

      val expected = GoldenModel.timestep(initState, spikeAddrs, wf, wx, p, v, aBar, bBar, bias, cfg)

      TestUtils.runTimestep(dut, cfg, spikeAddrs)

      // Verify membrane potentials
      for (i <- 0 until cfg.nNeurons) {
        val actual = dut.io.membraneReadout(i).peekInt().toInt
        assert(actual == expected.membrane(i),
          s"Neuron $i membrane mismatch: got $actual, expected ${expected.membrane(i)}")
      }

      // Verify spikes
      val (_, expectedSpikes) = GoldenModel.neuronUpdate(
        Array.fill(cfg.nNeurons)(0),
        GoldenModel.spikeIntegration(spikeAddrs, wf, cfg.nNeurons, cfg.accBits),
        GoldenModel.memoryIntegration(
          Array.fill(cfg.memDim)(0), p, v,
          GoldenModel.scalarDrive(spikeAddrs, wx, bias, cfg.accBits, cfg.mBits),
          cfg.nNeurons, cfg.accBits, cfg.internalAccBits
        ),
        cfg.beta, BETA_SHIFT, cfg.threshold, cfg.uBits, cfg.accBits
      )
      for (i <- 0 until cfg.nNeurons) {
        val actual = dut.io.outSpikes.vector(i).peekBoolean()
        assert(actual == expectedSpikes(i),
          s"Neuron $i spike mismatch: got $actual, expected ${expectedSpikes(i)}")
      }

      println("Test 1 PASSED: All membrane potentials and spikes match golden model")
    }
  }

  it should "evolve memory state correctly over multiple timesteps" in {
    test(new DmpCore(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      val wf = genWf(cfg.nNeurons, cfg.inputWidth)
      val wx = genWx(cfg.inputWidth)
      val p = genP(cfg.nNeurons, cfg.memDim)
      val v = genV(cfg.nNeurons)
      val aBar = genABar(cfg.memDim)
      val bBar = genBBar(cfg.memDim)
      val bias = 5

      TestUtils.loadAllWeights(dut, cfg, wf, wx, p, v, aBar, bBar, bias)

      val spikeSequence = Seq(
        Seq(0, 2),      // timestep 0
        Seq(1, 4, 7),   // timestep 1
        Seq(3)          // timestep 2
      )

      var state = GoldenModel.LayerState(
        membrane = Array.fill(cfg.nNeurons)(0),
        memory = Array.fill(cfg.memDim)(0)
      )

      for ((spikes, t) <- spikeSequence.zipWithIndex) {
        state = GoldenModel.timestep(state, spikes, wf, wx, p, v, aBar, bBar, bias, cfg)

        TestUtils.runTimestep(dut, cfg, spikes)
        dut.clock.step(1)

        for (i <- 0 until cfg.nNeurons) {
          val actual = dut.io.membraneReadout(i).peekInt().toInt
          assert(actual == state.membrane(i),
            s"Timestep $t, neuron $i membrane mismatch: got $actual, expected ${state.membrane(i)}")
        }
      }

      println("Test 2 PASSED: Memory state evolves correctly over 3 timesteps")
    }
  }

  it should "handle saturation boundaries correctly" in {
    test(new DmpCore(cfg)) { dut =>
      // Large weights to force overflow
      val wf = Array.tabulate(cfg.nNeurons, cfg.inputWidth)((_, _) => 127)
      val wx = Array.fill(cfg.inputWidth)(127)
      val p = Array.tabulate(cfg.nNeurons, cfg.memDim)((_, _) => 127)
      val v = Array.fill(cfg.nNeurons)(127)
      val aBar = Array.tabulate(cfg.memDim, cfg.memDim)((i, j) => if (i == j) 127 else 0)
      val bBar = Array.fill(cfg.memDim)(127)
      val bias = 100

      TestUtils.loadAllWeights(dut, cfg, wf, wx, p, v, aBar, bBar, bias)

      // 4 spikes to maximize accumulation
      val spikeAddrs = Seq(0, 1, 2, 3)
      val initState = GoldenModel.LayerState(
        membrane = Array.fill(cfg.nNeurons)(0),
        memory = Array.fill(cfg.memDim)(0)
      )

      val expected = GoldenModel.timestep(initState, spikeAddrs, wf, wx, p, v, aBar, bBar, bias, cfg)

      TestUtils.runTimestep(dut, cfg, spikeAddrs)

      for (i <- 0 until cfg.nNeurons) {
        val actual = dut.io.membraneReadout(i).peekInt().toInt
        assert(actual == expected.membrane(i),
          s"Saturation test, neuron $i: got $actual, expected ${expected.membrane(i)}")
      }

      println("Test 3 PASSED: Saturation handled correctly at boundaries")
    }
  }

  it should "produce correct spikes from threshold crossing" in {
    test(new DmpCore(cfg)) { dut =>
      // Design weights so specific neurons cross threshold (512)
      // With 2 spikes, each neuron gets Wf[i][addr0] + Wf[i][addr1]
      // Set Wf so neuron 0 and 1 get large values, others small
      val wf = Array.tabulate(cfg.nNeurons, cfg.inputWidth) { (i, j) =>
        if (i < 2 && j < 2) 127 else 1
      }
      val wx = Array.fill(cfg.inputWidth)(10)
      val p = Array.tabulate(cfg.nNeurons, cfg.memDim)((_, _) => 0)
      val v = Array.fill(cfg.nNeurons)(0)
      val aBar = Array.tabulate(cfg.memDim, cfg.memDim)((i, j) => if (i == j) 100 else 0)
      val bBar = Array.fill(cfg.memDim)(1)
      val bias = 5

      TestUtils.loadAllWeights(dut, cfg, wf, wx, p, v, aBar, bBar, bias)

      // Two timesteps: first builds up membrane, second may fire
      val spikes1 = Seq(0, 1)
      val initState = GoldenModel.LayerState(
        membrane = Array.fill(cfg.nNeurons)(0),
        memory = Array.fill(cfg.memDim)(0)
      )

      val state1 = GoldenModel.timestep(initState, spikes1, wf, wx, p, v, aBar, bBar, bias, cfg)
      TestUtils.runTimestep(dut, cfg, spikes1)
      dut.clock.step(1)

      // Second timestep to potentially push over threshold
      val state2 = GoldenModel.timestep(state1, spikes1, wf, wx, p, v, aBar, bBar, bias, cfg)
      TestUtils.runTimestep(dut, cfg, spikes1)

      // Verify which neurons fired
      val (_, expectedSpikes) = GoldenModel.neuronUpdate(
        state1.membrane,
        GoldenModel.spikeIntegration(spikes1, wf, cfg.nNeurons, cfg.accBits),
        GoldenModel.memoryIntegration(
          state1.memory, p, v,
          GoldenModel.scalarDrive(spikes1, wx, bias, cfg.accBits, cfg.mBits),
          cfg.nNeurons, cfg.accBits, cfg.internalAccBits
        ),
        cfg.beta, BETA_SHIFT, cfg.threshold, cfg.uBits, cfg.accBits
      )

      var firingCount = 0
      for (i <- 0 until cfg.nNeurons) {
        val actual = dut.io.outSpikes.vector(i).peekBoolean()
        assert(actual == expectedSpikes(i),
          s"Threshold test, neuron $i spike: got $actual, expected ${expectedSpikes(i)}")
        if (expectedSpikes(i)) firingCount += 1
      }

      // Also verify soft reset in membrane
      for (i <- 0 until cfg.nNeurons) {
        val actual = dut.io.membraneReadout(i).peekInt().toInt
        assert(actual == state2.membrane(i),
          s"Threshold test, neuron $i membrane after reset: got $actual, expected ${state2.membrane(i)}")
      }

      println(s"Test 4 PASSED: Threshold crossing correct, $firingCount neurons fired")
    }
  }

  it should "ReLU zeroes negative scalar drive" in {
    test(new DmpCore(cfg)) { dut =>
      // Wx weights all negative, bias small positive → net negative → x=0
      val wf = Array.tabulate(cfg.nNeurons, cfg.inputWidth)((i, j) => (i + j) % 3 + 1)
      val wx = Array.fill(cfg.inputWidth)(-20)  // Strongly negative
      val p = Array.tabulate(cfg.nNeurons, cfg.memDim)((_, _) => 50)
      val v = Array.fill(cfg.nNeurons)(50)
      val aBar = Array.tabulate(cfg.memDim, cfg.memDim)((i, j) => if (i == j) 100 else 0)
      val bBar = Array.fill(cfg.memDim)(10)
      val bias = 5  // Small positive bias, overwhelmed by negative weights

      TestUtils.loadAllWeights(dut, cfg, wf, wx, p, v, aBar, bBar, bias)

      val spikeAddrs = Seq(0, 1, 2)  // 3 spikes × (-20) + 5 = -55 → ReLU → 0
      val initState = GoldenModel.LayerState(
        membrane = Array.fill(cfg.nNeurons)(0),
        memory = Array.fill(cfg.memDim)(0)
      )

      val expected = GoldenModel.timestep(initState, spikeAddrs, wf, wx, p, v, aBar, bBar, bias, cfg)

      // Verify golden model: x should be 0
      val xExpected = GoldenModel.scalarDrive(spikeAddrs, wx, bias, cfg.accBits, cfg.mBits)
      assert(xExpected == 0, s"Golden model x should be 0 (ReLU), got $xExpected")

      TestUtils.runTimestep(dut, cfg, spikeAddrs)

      for (i <- 0 until cfg.nNeurons) {
        val actual = dut.io.membraneReadout(i).peekInt().toInt
        assert(actual == expected.membrane(i),
          s"ReLU test, neuron $i: got $actual, expected ${expected.membrane(i)}")
      }

      println("Test 5 PASSED: ReLU correctly zeroes negative scalar drive")
    }
  }

  it should "handle maximum spike count correctly" in {
    test(new DmpCore(cfg)) { dut =>
      val wf = genWf(cfg.nNeurons, cfg.inputWidth)
      val wx = genWx(cfg.inputWidth)
      val p = genP(cfg.nNeurons, cfg.memDim)
      val v = genV(cfg.nNeurons)
      val aBar = genABar(cfg.memDim)
      val bBar = genBBar(cfg.memDim)
      val bias = 2

      TestUtils.loadAllWeights(dut, cfg, wf, wx, p, v, aBar, bBar, bias)

      // Fill all maxSpikesPerStep addresses
      val spikeAddrs = (0 until cfg.maxSpikesPerStep).toSeq
      val initState = GoldenModel.LayerState(
        membrane = Array.fill(cfg.nNeurons)(0),
        memory = Array.fill(cfg.memDim)(0)
      )

      val expected = GoldenModel.timestep(initState, spikeAddrs, wf, wx, p, v, aBar, bBar, bias, cfg)

      TestUtils.runTimestep(dut, cfg, spikeAddrs)

      for (i <- 0 until cfg.nNeurons) {
        val actual = dut.io.membraneReadout(i).peekInt().toInt
        assert(actual == expected.membrane(i),
          s"Max spikes test, neuron $i: got $actual, expected ${expected.membrane(i)}")
      }

      println("Test 6 PASSED: Maximum spike count handled correctly")
    }
  }
}
