//! Memory-mapped register definitions for the Rockchip NPU.
//!
//! The register layout is described using [`tock_registers`], which provides a
//! safe and zero-cost abstraction over volatile MMIO access.  Each functional
//! block is exposed through a dedicated sub-module so the code that drives the
//! hardware can depend on a well-structured Rust API instead of scattering raw
//! offsets across the driver.

use core::{ops::Deref, ptr::NonNull};

use tock_registers::{interfaces::Readable, register_structs, registers::*};

pub mod consts;
pub mod global;
pub mod int;
pub mod pc;

register_structs! {
    pub RknpuRegistersRaw {
        (0x0000 => pub version: ReadOnly<u32>),
        (0x0004 => pub version_num: ReadOnly<u32>),
        (0x0008 => @END),
    }
}

/// Top-level view of the RKNPU register file.
///
/// The low-level register groups live in separate modules (`pc`, `int`,
/// `global`) and are re-exported here as zero-sized typed pointers into the
/// MMIO region. Users should create this from an MMIO base address and then
/// call the group accessors to obtain references to the specific register
/// blocks.
pub struct RknpuRegisters {
    base: NonNull<RknpuRegistersRaw>,
}
unsafe impl Send for RknpuRegisters {}

impl RknpuRegisters {
    /// Create a new facade over the RKNPU MMIO region.
    ///
    /// # Safety
    ///
    /// The caller must ensure the provided pointer is a valid mapping for the
    /// RKNPU register file for the lifetime of the returned object.
    pub const unsafe fn new(base_addr: NonNull<u8>) -> Self {
        Self {
            base: base_addr.cast(),
        }
    }

    pub fn version(&self) -> u32 {
        self.version.get() + self.version_num.get() & 0xffff
    }
}

impl Deref for RknpuRegisters {
    type Target = RknpuRegistersRaw;

    fn deref(&self) -> &Self::Target {
        unsafe { self.base.as_ref() }
    }
}
