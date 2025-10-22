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
mod gem;
mod job;
mod osal;
mod registers;
mod task;

use alloc::vec::Vec;
pub use config::*;
pub use err::*;
pub use gem::*;
pub use job::*;
pub use osal::*;
use rdif_base::DriverGeneric;
use spin::Mutex;
pub use task::*;
use tock_registers::interfaces::*;

use crate::{
    data::RknpuData,
    registers::{RknpuCore, consts::INT_CLEAR_ALL},
};

const VERSION_MAJOR: u32 = 0;
const VERSION_MINOR: u32 = 9;
const VERSION_PATCH: u32 = 8;
const RKNPU_PC_DATA_EXTRA_AMOUNT: u32 = 4;
const PC_INTERRUPT_VALID_MASK: u32 = (1 << 14) - 1;

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
    base: Vec<RknpuCore>,
    #[allow(dead_code)]
    config: RknpuConfig,
    data: RknpuData,
    iommu_enabled: bool,
    irq_lock: Mutex<()>,
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
                .map(|&addr| unsafe { RknpuCore::new(addr) })
                .collect(),
            data,
            config,
            iommu_enabled: false,
            irq_lock: Mutex::new(()),
        }
    }

    pub fn open(&mut self) -> Result<(), RknpuError> {
        Ok(())
    }

    #[allow(dead_code)]
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
            let pc_data_addr = self.base[0].pc().base_address.get();
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

    // /// Commit a prepared job descriptor to the hardware command parser.
    // pub fn commit_job(&mut self, job: &mut RknpuJob) -> Result<(), RknpuError> {
    //     self.job_commit(job)
    // }

    /// Busy-wait until the specified interrupt mask is observed for the given core.
    pub fn wait_for_completion(
        &mut self,
        core_idx: usize,
        mask: u32,
        timeout: usize,
    ) -> Result<(), RknpuError> {
        if mask == 0 {
            return Err(RknpuError::InvalidParameter);
        }

        let Some(base) = self.base.get(core_idx) else {
            return Err(RknpuError::InvalidParameter);
        };

        // Clear any stale status bits before we start polling.
        base.pc()
            .interrupt_clear
            .set(mask & PC_INTERRUPT_VALID_MASK);
        base.int().int_clear.set(mask & PC_INTERRUPT_VALID_MASK);

        const LOG_INTERVAL: usize = 100_000;
        let mut observed: u32 = 0;
        for iteration in 0..timeout {
            let raw_status = base.pc().interrupt_raw_status.get();
            let valid_status = raw_status & PC_INTERRUPT_VALID_MASK;

            if valid_status != 0 {
                observed |= valid_status;
                // Acknowledge the interrupt sources so the hardware can
                // continue progressing through the pipeline.
                base.pc()
                    .interrupt_clear
                    .set(INT_CLEAR_ALL & PC_INTERRUPT_VALID_MASK);
                base.int()
                    .int_clear
                    .set(INT_CLEAR_ALL & PC_INTERRUPT_VALID_MASK);

                if observed & mask == mask {
                    return Ok(());
                }
            }

            if iteration % LOG_INTERVAL == 0 {
                let status = base.pc().interrupt_status.get();
                let global_status = base.int().int_status.get();
                let global_raw = base.int().int_raw_status.get();
                let enable = base.pc().operation_enable.get();
                let task_status = base.pc().task_status.get();
                let pc_mask = base.pc().interrupt_mask.get();
                let pc_base = base.pc().base_address.get();
                let reg_amounts = base.pc().register_amounts.get();
                let task_control = base.pc().task_control.get();
                let task_dma = base.pc().task_dma_base_addr.get();
                let global_mask = base.int().int_mask.get();
                let global_enable = base.global().enable_mask.get();
                debug!(
                    "wait_for_completion[core={}]: iter={} status=0x{:x} raw=0x{:x} pc_mask=0x{:x} pc_base=0x{:x} reg_amounts=0x{:x} task_ctrl=0x{:x} task_dma=0x{:x} op_enable=0x{:x} task_status=0x{:x} global_mask=0x{:x} global_status=0x{:x} global_raw=0x{:x} global_enable=0x{:x}",
                    core_idx,
                    iteration,
                    status,
                    raw_status,
                    pc_mask,
                    pc_base,
                    reg_amounts,
                    task_control,
                    task_dma,
                    enable,
                    task_status,
                    global_mask,
                    global_status,
                    global_raw,
                    global_enable
                );
            }

            core::hint::spin_loop();
        }

        let final_status = base.pc().interrupt_status.get();
        let final_raw = base.pc().interrupt_raw_status.get();
        let final_global_status = base.int().int_status.get();
        let final_global_raw = base.int().int_raw_status.get();
        let final_enable = base.pc().operation_enable.get();
        let final_task_status = base.pc().task_status.get();
        let final_pc_mask = base.pc().interrupt_mask.get();
        let final_pc_base = base.pc().base_address.get();
        let final_reg_amounts = base.pc().register_amounts.get();
        let final_task_control = base.pc().task_control.get();
        let final_task_dma = base.pc().task_dma_base_addr.get();
        let final_global_mask = base.int().int_mask.get();
        let final_global_enable = base.global().enable_mask.get();
        error!(
            "wait_for_completion timeout: core={} mask=0x{:x} status=0x{:x} raw=0x{:x} pc_mask=0x{:x} pc_base=0x{:x} reg_amounts=0x{:x} task_ctrl=0x{:x} task_dma=0x{:x} op_enable=0x{:x} task_status=0x{:x} global_mask=0x{:x} global_status=0x{:x} global_raw=0x{:x} global_enable=0x{:x}",
            core_idx,
            mask,
            final_status,
            final_raw,
            final_pc_mask,
            final_pc_base,
            final_reg_amounts,
            final_task_control,
            final_task_dma,
            final_enable,
            final_task_status,
            final_global_mask,
            final_global_status,
            final_global_raw,
            final_global_enable
        );

        Err(RknpuError::Timeout)
    }

    // fn job_commit(&mut self, job: &mut RknpuJob) -> Result<(), RknpuError> {
    //     const CORE0_1_MASK: u32 = RKNPU_CORE0_MASK | RKNPU_CORE1_MASK;
    //     const CORE0_1_2_MASK: u32 = RKNPU_CORE0_MASK | RKNPU_CORE1_MASK | RKNPU_CORE2_MASK;

    //     match job.args.core_mask {
    //         RKNPU_CORE0_MASK => self.sub_core_submit(job, 0)?,
    //         RKNPU_CORE1_MASK => self.sub_core_submit(job, 1)?,
    //         RKNPU_CORE2_MASK => self.sub_core_submit(job, 2)?,
    //         CORE0_1_MASK => {
    //             self.sub_core_submit(job, 0)?;
    //             self.sub_core_submit(job, 1)?;
    //         }
    //         CORE0_1_2_MASK => {
    //             self.sub_core_submit(job, 0)?;
    //             self.sub_core_submit(job, 1)?;
    //             self.sub_core_submit(job, 2)?;
    //         }
    //         _ => {
    //             error!("Invalid core mask: 0x{:x}", job.args.core_mask);
    //         }
    //     }

    //     Ok(())
    // }

    pub fn submit(&mut self, job: &mut RknpuSubmitK, core_idx: usize) -> Result<(), RknpuError> {
        job.tasks.confirm_write_all();
        self.base[core_idx].submit(
            &self.data,
            job.flags,
            job.tasks.as_ref(),
            job.tasks.bus_addr() as _,
            core_idx,
        )?;

        Ok(())
    }
}

impl DriverGeneric for Rknpu {
    fn open(&mut self) -> Result<(), rdif_base::KError> {
        Self::open(self).map_err(|_| rdif_base::KError::Unknown("open fail"))
    }

    fn close(&mut self) -> Result<(), rdif_base::KError> {
        Ok(())
    }
}

/// Selects the lowest-numbered available core and returns its mask.
#[allow(dead_code)]
fn select_first_core(available_mask: u32) -> Option<u32> {
    (0..RKNPU_MAX_CORES)
        .map(core_mask_from_index)
        .find(|mask| available_mask & *mask != 0)
}
