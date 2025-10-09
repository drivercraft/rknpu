//! Low-level building blocks for interacting with Rockchip NPUs.
//!
//! The crate keeps the register descriptions and helper types in Rust so that
//! higher-level driver components can avoid hard-coded offsets and rely on type
//! safe accessors instead.
//!
//! This implementation provides a minimal device layer with OSAL abstractions
//! for platform-independent hardware operations.

#![no_std]

use core::ptr::NonNull;

extern crate alloc;

mod config;
mod data;
mod err;
mod osal;
mod registers;
// mod hal;
// mod memory;
// mod device;

use alloc::vec::Vec;
pub use config::*;
pub use err::*;
pub use osal::*;

use crate::{data::RknpuData, registers::RknpuRegisters};

pub struct Rknpu {
    base: Vec<RknpuRegisters>,
    config: RknpuConfig,
    data: RknpuData,
    iommu_enabled: bool,
}

impl Rknpu {
    /// Creates a new RKNPU interface from a raw MMIO base address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `base_addr` is the correctly mapped and
    /// aligned physical address of the RKNPU register file and that it remains
    /// valid for the lifetime of the returned structure.
    pub fn new(base_addrs: &[NonNull<u8>], config: RknpuConfig) -> Self {
        let data = RknpuData::new(config.rknpu_type);

        Self {
            base: base_addrs
                .iter()
                .map(|&addr| unsafe { RknpuRegisters::new(addr) })
                .collect(),
            data: RknpuData::new(config.rknpu_type),
            config,
            iommu_enabled: false,
        }
    }

    pub fn open(&mut self) -> Result<(), RknpuError> {
        Ok(())
    }

    fn dma_bit_mask(&self) -> u64 {
        self.data.dma_mask
    }

    pub fn get_hw_version(&self) -> u32 {
        self.base[0].version()
    }
}
