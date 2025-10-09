//! Low-level building blocks for interacting with Rockchip NPUs.
//!
//! The crate keeps the register descriptions and helper types in Rust so that
//! higher-level driver components can avoid hard-coded offsets and rely on type
//! safe accessors instead.
//!
//! This implementation provides a minimal device layer with OSAL abstractions
//! for platform-independent hardware operations.

#![no_std]

extern crate alloc;
#[macro_use]
extern crate log;

use core::ptr::NonNull;

mod config;
mod data;
mod err;
mod osal;
mod registers;

use alloc::vec::Vec;
pub use config::*;
pub use err::*;
pub use osal::*;
use tock_registers::interfaces::*;

use crate::{data::RknpuData, registers::RknpuRegisters};

const VERSION_MAJOR: u32 = 0;
const VERSION_MINOR: u32 = 9;
const VERSION_PATCH: u32 = 8;

const fn version(major: u32, minor: u32, patch: u32) -> u32 {
    major * 10000 + minor * 100 + patch
}

/// Action flags for RKNPU operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RknpuAction {
    GetHwVersion = 0,
    GetDrvVersion = 1,
    GetFreq = 2,
    SetFreq = 3,
    GetVolt = 4,
    SetVolt = 5,
    ActReset = 6,
    GetBwPriority = 7,
    SetBwPriority = 8,
    GetBwExpect = 9,
    SetBwExpect = 10,
    GetBwTw = 11,
    SetBwTw = 12,
    ActClrTotalRwAmount = 13,
    GetDtWrAmount = 14,
    GetDtRdAmount = 15,
    GetWtRdAmount = 16,
    GetTotalRwAmount = 17,
    GetIommuEn = 18,
    SetProcNice = 19,
    PowerOn = 20,
    PowerOff = 21,
    GetTotalSramSize = 22,
    GetFreeSramSize = 23,
    GetIommuDomainId = 24,
    SetIommuDomainId = 25,
}

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
            data,
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

    pub fn clear_rw_amount(&mut self) -> Result<(), RknpuError> {
        let Some(amount_top) = self.data.amount_top else {
            warn!("RKNPU does not support read/write amount statistics");
            return Ok(());
        };

        if self.data.pc_dma_ctrl > 0 {
            let pc_data_addr = self.base[0].pc().regs().base_address.get();
            unsafe {
                self.base[0]
                    .offset_ptr::<u32>(pc_data_addr as usize)
                    .write_volatile(1);
                self.base[0]
                    .offset_ptr::<u32>(amount_top.offset_clr_all as usize)
                    .write_volatile(0x80000101);
                self.base[0]
                    .offset_ptr::<u32>(amount_top.offset_clr_all as usize)
                    .write_volatile(0x00000101);
                if let Some(amount_core) = self.data.amount_core {
                    self.base[0]
                        .offset_ptr::<u32>(amount_core.offset_clr_all as usize)
                        .write_volatile(0x80000101);
                    self.base[0]
                        .offset_ptr::<u32>(amount_core.offset_clr_all as usize)
                        .write_volatile(0x00000101);
                }
            };
        } else {
            unsafe {
                self.base[0]
                    .offset_ptr::<u32>(amount_top.offset_clr_all as usize)
                    .write_volatile(0x80000101);
                self.base[0]
                    .offset_ptr::<u32>(amount_top.offset_clr_all as usize)
                    .write_volatile(0x00000101);
                if let Some(amount_core) = self.data.amount_core {
                    self.base[0]
                        .offset_ptr::<u32>(amount_core.offset_clr_all as usize)
                        .write_volatile(0x80000101);
                    self.base[0]
                        .offset_ptr::<u32>(amount_core.offset_clr_all as usize)
                        .write_volatile(0x00000101);
                }
            }
        }

        Ok(())
    }

    /// Execute an RKNPU action based on the provided action request
    ///
    /// This function mirrors the Linux driver's rknpu_action implementation,
    /// providing a Rust-safe interface for hardware operations.
    pub fn action(&mut self, action: RknpuAction) -> Result<u32, RknpuError> {
        match action {
            RknpuAction::GetHwVersion => {
                let val = self.get_hw_version();
                Ok(val)
            }
            RknpuAction::GetDrvVersion => Ok(version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)),
            RknpuAction::GetFreq => {
                // TODO FPGA频率获取
                Ok(0)
            }
            RknpuAction::SetFreq => {
                // 频率设置 - 需要时钟管理
                Ok(0)
            }
            RknpuAction::GetVolt => {
                // TODO FPGA电压获取
                Ok(0)
            }
            RknpuAction::SetVolt => Err(RknpuError::InternalError),
            RknpuAction::ActReset => {
                // TODO FPGA复位操作
                Ok(0)
            }
            RknpuAction::GetBwPriority => {
                // 带宽优先级获取
                Err(RknpuError::InternalError)
            }
            RknpuAction::SetBwPriority => {
                // 带宽优先级设置
                log::warn!("SetBwPriority operation not yet implemented");
                Err(RknpuError::InternalError)
            }
            RknpuAction::GetBwExpect => {
                // 带宽期望值获取
                Err(RknpuError::InternalError)
            }
            RknpuAction::SetBwExpect => {
                // 带宽期望值设置
                log::warn!("SetBwExpect operation not yet implemented");
                Err(RknpuError::InternalError)
            }
            RknpuAction::GetBwTw => {
                // 带宽时间窗口获取
                Err(RknpuError::InternalError)
            }
            RknpuAction::SetBwTw => {
                // 带宽时间窗口设置
                Err(RknpuError::InternalError)
            }
            RknpuAction::ActClrTotalRwAmount => {
                // 清除读写总量统计
                self.clear_rw_amount()?;
                Ok(0)
            }
            RknpuAction::GetDtWrAmount => {
                // 获取设备写数据量
                warn!("Get rw_amount is not supported on this device!");
                Ok(0)
            }
            RknpuAction::GetDtRdAmount => {
                // 获取设备读数据量
                warn!("Get rw_amount is not supported on this device!");
                Ok(0)
            }
            RknpuAction::GetWtRdAmount => {
                // 获取等待读数据量
                warn!("Get rw_amount is not supported on this device!");
                Ok(0)
            }
            RknpuAction::GetTotalRwAmount => {
                // 获取总读写数据量
                warn!("Get rw_amount is not supported on this device!");
                Ok(0)
            }
            RknpuAction::GetIommuEn => {
                // 获取IOMMU启用状态
                Ok(if self.iommu_enabled { 1 } else { 0 })
            }
            RknpuAction::SetProcNice => {
                // 设置进程优先级 - 在内核空间不适用
                log::warn!("SetProcNice operation not applicable in bare metal context");
                Ok(0)
            }
            RknpuAction::PowerOn => {
                // 电源开启
                log::warn!("PowerOn operation not yet implemented");
                Ok(0)
            }
            RknpuAction::PowerOff => {
                // 电源关闭
                log::warn!("PowerOff operation not yet implemented");
                Ok(0)
            }
            RknpuAction::GetTotalSramSize => {
                // 获取总SRAM大小
                Ok(0)
            }
            RknpuAction::GetFreeSramSize => Ok(self.data.nbuf_size as u32),
            RknpuAction::GetIommuDomainId => {
                // 获取IOMMU域ID - 需要IOMMU管理
                log::warn!("GetIommuDomainId operation not yet implemented");
                Ok(0)
            }
            RknpuAction::SetIommuDomainId => {
                // 设置IOMMU域ID - 需要IOMMU管理
                // log::warn!("SetIommuDomainId operation not yet implemented");
                Ok(0)
            }
        }
    }

    /// Convenience method to check IOMMU status using action interface
    pub fn is_iommu_enabled(&self) -> bool {
        self.iommu_enabled
    }

    /// Enable or disable IOMMU
    pub fn set_iommu_enabled(&mut self, enabled: bool) {
        self.iommu_enabled = enabled;
    }
}
