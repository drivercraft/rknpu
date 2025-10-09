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

use alloc::vec::Vec;
pub use config::*;
pub use err::*;
pub use gem::*;
pub use job::*;
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
    #[allow(dead_code)]
    config: RknpuConfig,
    data: RknpuData,
    iommu_enabled: bool,
    gem: RknpuGemManager,
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
            gem: RknpuGemManager::new(),
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

    pub fn gem_manager(&self) -> &RknpuGemManager {
        &self.gem
    }

    pub fn gem_manager_mut(&mut self) -> &mut RknpuGemManager {
        &mut self.gem
    }

    /// Submit an inference workload to the hardware queue.
    ///
    /// The Rust port keeps the validation and bookkeeping behaviour from the
    /// upstream C driver while replacing the blocking wait queues, DMA fences
    /// and kernel list management with no-op placeholders so it can build in a
    /// bare-metal context.  Unsupported features (such as explicit fence
    /// handling) return `NotSupported` to mirror the `.config` options which
    /// are disabled in this environment.
    pub fn submit(&mut self, submit: &mut RknpuSubmit) -> Result<(), RknpuError> {
        if submit.task_number == 0 {
            return Err(RknpuError::InvalidParameter);
        }

        if submit.flags & (RKNPU_JOB_FENCE_IN | RKNPU_JOB_FENCE_OUT) != 0 {
            // Fence support is gated behind CONFIG_ROCKCHIP_RKNPU_FENCE which
            // is not enabled by the supplied kernel configuration.
            return Err(RknpuError::NotSupported);
        }

        if !submit.is_pc_mode() {
            // The PC (program counter) execution mode is the only flow kept in
            // the minimal Rust implementation.
            return Err(RknpuError::NotSupported);
        }

        let available_mask = self.data.core_mask;
        let mut selected_mask = submit.core_mask;

        if selected_mask == RKNPU_CORE_AUTO_MASK {
            selected_mask = select_first_core(available_mask).ok_or(RknpuError::DeviceNotReady)?;
        }

        if selected_mask == 0 {
            return Err(RknpuError::InvalidParameter);
        }

        if selected_mask & !available_mask != 0 {
            return Err(RknpuError::InvalidParameter);
        }

        let task_handle =
            RknpuGemHandle::from_raw(submit.task_obj_addr).ok_or(RknpuError::InvalidHandle)?;

        let tasks = self
            .gem
            .task_slice(task_handle)
            .ok_or(RknpuError::InvalidHandle)?;

        let start = submit.task_start as usize;
        let end = start
            .checked_add(submit.task_number as usize)
            .ok_or(RknpuError::InvalidParameter)?;

        if end > tasks.len() {
            return Err(RknpuError::InvalidParameter);
        }

        let selected_tasks = &tasks[start..end];
        let last_task = selected_tasks.last().ok_or(RknpuError::InvalidParameter)?;

        // For now we execute the job immediately instead of queueing it on a
        // per-core work list.  Future OS bindings can hook into the placeholder
        // and provide real scheduling.
        let _placeholder_token = os_placeholder_schedule(selected_mask, submit.task_number);

        submit.core_mask = selected_mask;
        submit.task_counter = submit.task_number;
        submit.hw_elapse_time = 0;
        submit.task_base_addr = last_task.regcmd_addr;

        Ok(())
    }
}

/// Selects the lowest-numbered available core and returns its mask.
fn select_first_core(available_mask: u32) -> Option<u32> {
    (0..RKNPU_MAX_CORES)
        .map(core_mask_from_index)
        .find(|mask| available_mask & *mask != 0)
}

/// Placeholder representing the OS specific scheduling hook.
fn os_placeholder_schedule(core_mask: u32, task_number: u32) -> CoreScheduleToken {
    let _ = (core_mask, task_number);
    CoreScheduleToken {}
}

/// Zero-sized token returned by the scheduling placeholder.
struct CoreScheduleToken {}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::ptr::NonNull;

    use super::*;

    fn dummy_addr() -> NonNull<u8> {
        static mut BYTE: u8 = 0;
        unsafe { NonNull::new_unchecked((&mut BYTE) as *mut u8) }
    }

    fn test_rknpu() -> Rknpu {
        let base = [dummy_addr()];
        let config = RknpuConfig {
            rknpu_type: RknpuType::Rk3588,
        };
        Rknpu::new(&base, config)
    }

    #[test]
    fn reject_zero_tasks() {
        let mut rknpu = test_rknpu();
        let mut submit = RknpuSubmit {
            task_number: 0,
            ..Default::default()
        };

        let err = rknpu.submit(&mut submit).unwrap_err();
        assert_eq!(err, RknpuError::InvalidParameter);
    }

    #[test]
    fn require_pc_flag() {
        let mut rknpu = test_rknpu();
        let mut submit = RknpuSubmit {
            task_number: 1,
            ..Default::default()
        };

        let err = rknpu.submit(&mut submit).unwrap_err();
        assert_eq!(err, RknpuError::NotSupported);
    }

    #[test]
    fn happy_path_sets_defaults() {
        let mut rknpu = test_rknpu();
        let tasks = (0..4)
            .map(|idx| RknpuTask {
                regcmd_addr: 0x1000 + idx as u64 * 0x40,
                ..RknpuTask::default()
            })
            .collect();
        let handle = rknpu
            .gem_manager_mut()
            .create_from_tasks(tasks, 0, 0, RKNPU_CORE0_MASK)
            .expect("failed to create task buffer");
        let mut submit = RknpuSubmit {
            flags: RKNPU_JOB_PC,
            task_number: 4,
            ..Default::default()
        };
        submit.task_obj_addr = handle.as_raw();

        rknpu.submit(&mut submit).unwrap();

        assert_eq!(submit.task_counter, 4);
        assert_eq!(submit.core_mask, RKNPU_CORE0_MASK);
        assert_eq!(submit.hw_elapse_time, 0);
        assert_eq!(submit.task_base_addr, 0x1000 + 3 * 0x40);
    }

    #[test]
    fn reject_missing_task_buffer() {
        let mut rknpu = test_rknpu();
        let mut submit = RknpuSubmit {
            flags: RKNPU_JOB_PC,
            task_number: 2,
            ..Default::default()
        };
        let err = rknpu.submit(&mut submit).unwrap_err();
        assert_eq!(err, RknpuError::InvalidHandle);
    }
}
