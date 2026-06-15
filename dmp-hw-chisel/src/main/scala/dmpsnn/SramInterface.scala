package dmpsnn

import chisel3._
import chisel3.util._

class SramReadPort(val addrWidth: Int, val dataWidth: Int) extends Bundle {
  val en   = Input(Bool())
  val addr = Input(UInt(addrWidth.W))
  val data = Output(UInt(dataWidth.W))
}

class SramWritePort(val addrWidth: Int, val dataWidth: Int) extends Bundle {
  val en   = Input(Bool())
  val addr = Input(UInt(addrWidth.W))
  val data = Input(UInt(dataWidth.W))
}

/** Single-port synchronous SRAM with 1-cycle read latency.
  * Synthesizes to FPGA block RAM or ASIC SRAM macro.
  */
class Sram(val depth: Int, val width: Int) extends Module {
  val addrWidth = log2Ceil(math.max(depth, 2))

  val io = IO(new Bundle {
    val en    = Input(Bool())
    val wen   = Input(Bool())
    val addr  = Input(UInt(addrWidth.W))
    val wdata = Input(UInt(width.W))
    val rdata = Output(UInt(width.W))
  })

  val mem = SyncReadMem(depth, UInt(width.W))

  io.rdata := DontCare

  when(io.en) {
    when(io.wen) {
      mem.write(io.addr, io.wdata)
    }.otherwise {
      io.rdata := mem.read(io.addr)
    }
  }
}

/** Wide SRAM: `depth` addressable rows, each row is a Vec of `vecWidth` elements.
  * Single address yields all `vecWidth` words in parallel (one cycle latency).
  * Backed by a single SyncReadMem of Vec type.
  */
class WideSram(val depth: Int, val wordWidth: Int, val vecWidth: Int) extends Module {
  val addrWidth = log2Ceil(math.max(depth, 2))

  val io = IO(new Bundle {
    val en    = Input(Bool())
    val wen   = Input(Bool())
    val addr  = Input(UInt(addrWidth.W))
    val wdata = Input(Vec(vecWidth, UInt(wordWidth.W)))
    val rdata = Output(Vec(vecWidth, UInt(wordWidth.W)))
  })

  val mem = SyncReadMem(depth, Vec(vecWidth, UInt(wordWidth.W)))

  io.rdata := DontCare

  when(io.en) {
    when(io.wen) {
      mem.write(io.addr, io.wdata)
    }.otherwise {
      io.rdata := mem.read(io.addr)
    }
  }
}
