//! Memory-mapped register definitions for the Rockchip NPU.
//!
//! The register layout is described using [`tock_registers`], which provides a
//! safe and zero-cost abstraction over volatile MMIO access.  Each functional
//! block is exposed through a dedicated sub-module so the code that drives the
//! hardware can depend on a well-structured Rust API instead of scattering raw
//! offsets across the driver.

use core::ptr::NonNull;

pub mod consts;
pub mod global;
pub mod int;
pub mod pc;

/// Top-level view of the RKNPU register file.
///
/// The low-level register groups live in separate modules (`pc`, `int`,
/// `global`) and are re-exported here as zero-sized typed pointers into the
/// MMIO region. Users should create this from an MMIO base address and then
/// call the group accessors to obtain references to the specific register
/// blocks.
pub struct RknpuRegisters {
    base: NonNull<u8>,
}

impl RknpuRegisters {
    /// Create a new facade over the RKNPU MMIO region.
    ///
    /// # Safety
    ///
    /// The caller must ensure the provided pointer is a valid mapping for the
    /// RKNPU register file for the lifetime of the returned object.
    pub const unsafe fn new(base_addr: NonNull<u8>) -> Self {
        Self { base: base_addr }
    }

    /// Returns a pointer to the PC register group mapped at the base.
    pub fn pc(&self) -> pc::PcRegisters {
        unsafe { pc::PcRegisters::from_base(self.base) }
    }

    /// Returns a pointer to the interrupt register group.
    pub fn int(&self) -> int::IntRegisters {
        unsafe { int::IntRegisters::from_base(self.base) }
    }

    /// Returns a pointer to global registers (enable mask etc.).
    pub fn global(&self) -> global::GlobalRegisters {
        unsafe { global::GlobalRegisters::from_base(self.base) }
    }
}
