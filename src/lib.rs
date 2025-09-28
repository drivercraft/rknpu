//! Low-level building blocks for interacting with Rockchip NPUs.
//!
//! The crate keeps the register descriptions and helper types in Rust so that
//! higher-level driver components can avoid hard-coded offsets and rely on type
//! safe accessors instead.

#![no_std]

use core::ptr::NonNull;

extern crate alloc;

mod config;
mod err;
mod registers;

pub use config::*;
pub use err::*;

pub struct Rknpu {
    reg: registers::RknpuRegisters,
    config: RknpuConfig,
    iommu_enabled: bool,
}

unsafe impl Send for Rknpu {}

impl Rknpu {
    /// Creates a new RKNPU interface from a raw MMIO base address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `base_addr` is the correctly mapped and
    /// aligned physical address of the RKNPU register file and that it remains
    /// valid for the lifetime of the returned structure.
    pub fn new(base_addr: NonNull<u8>, config: RknpuConfig) -> Self {
        Self {
            reg: unsafe { registers::RknpuRegisters::new(base_addr) },
            config,
            iommu_enabled: false,
        }
    }

    pub fn open(&mut self) -> Result<(), RknpuError> {
        Ok(())
    }

    fn dma_bit_mask(&self) -> u64 {
        self.config.dma_mask
    }
}
