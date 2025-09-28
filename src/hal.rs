//! Hardware abstraction layer for RKNPU device operations
//! 
//! This module implements the core hardware operation logic, including
//! task submission, memory management, and device control.

use core::ptr::NonNull;
use alloc::vec::Vec;
use crate::osal::{Osal, OsalError, MemoryBuffer, MemoryFlags, DmaSyncDirection, TimeStamp};
use crate::registers::RknpuRegisters;
use crate::config::RknpuConfig;
use crate::err::RknpuError;

/// Task execution flags
#[derive(Debug, Clone, Copy)]
pub struct TaskFlags {
    pub pc_mode: bool,
    pub non_block: bool,
    pub ping_pong: bool,
}

impl Default for TaskFlags {
    fn default() -> Self {
        Self {
            pc_mode: true,
            non_block: false,
            ping_pong: false,
        }
    }
}

/// Single NPU task descriptor
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct RknpuTask {
    pub flags: u32,
    pub op_idx: u32,
    pub enable_mask: u32,
    pub int_mask: u32,
    pub int_clear: u32,
    pub int_status: u32,
    pub regcfg_amount: u32,
    pub regcfg_offset: u32,
    pub regcmd_addr: u64,
}

impl Default for RknpuTask {
    fn default() -> Self {
        Self {
            flags: 0,
            op_idx: 0,
            enable_mask: 0,
            int_mask: 0,
            int_clear: 0,
            int_status: 0,
            regcfg_amount: 0,
            regcfg_offset: 0,
            regcmd_addr: 0,
        }
    }
}

/// Subcore task configuration for multi-core systems
#[derive(Debug, Clone, Copy)]
pub struct SubcoreTask {
    pub task_start: u32,
    pub task_number: u32,
}

impl Default for SubcoreTask {
    fn default() -> Self {
        Self {
            task_start: 0,
            task_number: 0,
        }
    }
}

/// Task submission parameters
#[derive(Debug, Clone)]
pub struct TaskSubmission {
    pub flags: TaskFlags,
    pub timeout_ms: u32,
    pub task_start: u32,
    pub task_number: u32,
    pub priority: i32,
    pub task_buffer: MemoryBuffer,
    pub task_base_addr: u64,
    pub core_mask: u32,
    pub subcore_tasks: Vec<SubcoreTask>,
}

/// Hardware job state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    Idle,
    Pending,
    Running,
    Completed,
    Error,
    Timeout,
}

/// Hardware job descriptor
#[derive(Debug)]
pub struct HardwareJob {
    pub id: u32,
    pub state: JobState,
    pub submission: TaskSubmission,
    pub start_time: TimeStamp,
    pub hw_start_time: TimeStamp,
    pub hw_end_time: TimeStamp,
    pub first_task: Option<RknpuTask>,
    pub last_task: Option<RknpuTask>,
    pub int_status: Vec<u32>,
    pub core_index: usize,
}

impl HardwareJob {
    pub fn new(id: u32, submission: TaskSubmission, core_index: usize) -> Self {
        let num_cores = submission.subcore_tasks.len().max(1);
        Self {
            id,
            state: JobState::Pending,
            submission,
            start_time: 0,
            hw_start_time: 0,
            hw_end_time: 0,
            first_task: None,
            last_task: None,
            int_status: vec![0; num_cores],
            core_index,
        }
    }
    
    pub fn elapsed_time_us(&self, current_time: TimeStamp) -> u64 {
        if self.start_time == 0 {
            0
        } else {
            current_time.saturating_sub(self.start_time)
        }
    }
    
    pub fn hw_elapsed_time_us(&self) -> u64 {
        if self.hw_start_time == 0 || self.hw_end_time == 0 {
            0
        } else {
            self.hw_end_time.saturating_sub(self.hw_start_time)
        }
    }
}

/// Hardware device state management
#[derive(Debug)]
pub struct DeviceState {
    pub power_on: bool,
    pub reset_in_progress: bool,
    pub iommu_enabled: bool,
    pub clock_enabled: bool,
    pub job_queue: Vec<HardwareJob>,
    pub next_job_id: u32,
}

impl Default for DeviceState {
    fn default() -> Self {
        Self {
            power_on: false,
            reset_in_progress: false,
            iommu_enabled: false,
            clock_enabled: false,
            job_queue: Vec::new(),
            next_job_id: 1,
        }
    }
}

/// Core mask definitions for different cores
pub const RKNPU_CORE0_MASK: u32 = 0x1;
pub const RKNPU_CORE1_MASK: u32 = 0x2;
pub const RKNPU_CORE2_MASK: u32 = 0x4;
pub const RKNPU_CORE_AUTO_MASK: u32 = 0x0;

/// PC data extra amount constant
pub const PC_DATA_EXTRA_AMOUNT: u32 = 4;

/// Hardware abstraction layer implementation
pub struct HardwareLayer<O: Osal> {
    pub registers: RknpuRegisters,
    pub config: RknpuConfig,
    pub osal: O,
    pub state: DeviceState,
    pub base_addrs: Vec<NonNull<u8>>,
}

impl<O: Osal> HardwareLayer<O> {
    /// Create new hardware layer instance
    pub fn new(base_addrs: Vec<NonNull<u8>>, config: RknpuConfig, osal: O) -> Result<Self, RknpuError> {
        if base_addrs.is_empty() {
            return Err(RknpuError::InvalidParameter);
        }
        
        let registers = unsafe { RknpuRegisters::new(base_addrs[0]) };
        
        Ok(Self {
            registers,
            config,
            osal,
            state: DeviceState::default(),
            base_addrs,
        })
    }
    
    /// Initialize hardware device
    pub fn initialize(&mut self) -> Result<(), RknpuError> {
        self.osal.log_info("Initializing RKNPU hardware");
        
        // Check hardware version
        let version = self.get_hw_version()?;
        self.osal.log_info(&alloc::format!("RKNPU HW version: 0x{:x}", version));
        
        // Initialize device state if needed
        if let Some(state_init) = self.config.state_init {
            self.osal.log_debug("Running platform-specific state initialization");
            state_init(self)?;
        }
        
        self.state.power_on = true;
        self.osal.log_info("RKNPU hardware initialized successfully");
        
        Ok(())
    }
    
    /// Get hardware version
    pub fn get_hw_version(&self) -> Result<u32, RknpuError> {
        // Read version from register offset 0x0
        let version_reg = unsafe { 
            core::ptr::read_volatile((self.base_addrs[0].as_ptr() as *const u32).add(0x0 / 4))
        };
        Ok(version_reg)
    }
    
    /// Perform hardware soft reset
    pub fn soft_reset(&mut self) -> Result<(), RknpuError> {
        self.osal.log_info("Performing RKNPU soft reset");
        
        self.state.reset_in_progress = true;
        
        // Wake up any pending operations
        self.cancel_all_jobs();
        
        // Simulate reset delay
        self.osal.msleep(100);
        
        // Reset sequence would be implemented here with actual reset controls
        self.osal.udelay(10);
        
        self.state.reset_in_progress = false;
        
        // Re-initialize state if needed
        if let Some(state_init) = self.config.state_init {
            state_init(self)?;
        }
        
        self.osal.log_info("RKNPU soft reset completed");
        Ok(())
    }
    
    /// Submit task to hardware
    pub fn submit_task(&mut self, submission: TaskSubmission) -> Result<u32, RknpuError> {
        if self.state.reset_in_progress {
            return Err(RknpuError::DeviceBusy);
        }
        
        let core_index = self.get_core_index_from_mask(submission.core_mask);
        let job_id = self.state.next_job_id;
        self.state.next_job_id += 1;
        
        let mut job = HardwareJob::new(job_id, submission, core_index);
        job.start_time = self.osal.get_time_us();
        job.state = JobState::Running;
        
        self.osal.log_debug(&alloc::format!("Submitting job {} to core {}", job_id, core_index));
        
        // Commit job to hardware
        self.commit_job_to_core(&mut job)?;
        
        // For blocking submission, wait for completion
        if !job.submission.flags.non_block {
            self.wait_for_job_completion(&mut job)?;
        }
        
        self.state.job_queue.push(job);
        Ok(job_id)
    }
    
    /// Wait for job completion
    pub fn wait_for_job_completion(&mut self, job: &mut HardwareJob) -> Result<(), RknpuError> {
        let timeout_us = job.submission.timeout_ms * 1000;
        let start_time = self.osal.get_time_us();
        
        while job.state == JobState::Running {
            if self.osal.timeout_check(start_time, timeout_us) {
                job.state = JobState::Timeout;
                self.osal.log_error(&alloc::format!("Job {} timeout after {}ms", job.id, job.submission.timeout_ms));
                return Err(RknpuError::Timeout);
            }
            
            // Check hardware status (simplified)
            if self.check_job_completion(job)? {
                job.state = JobState::Completed;
                job.hw_end_time = self.osal.get_time_us();
                break;
            }
            
            self.osal.udelay(1000); // Check every 1ms
        }
        
        Ok(())
    }
    
    /// Commit job to specific core
    fn commit_job_to_core(&mut self, job: &mut HardwareJob) -> Result<(), RknpuError> {
        if !job.submission.flags.pc_mode {
            return Err(RknpuError::NotSupported);
        }
        
        self.commit_pc_job(job)
    }
    
    /// Commit PC mode job to hardware
    fn commit_pc_job(&mut self, job: &mut HardwareJob) -> Result<(), RknpuError> {
        let core_index = job.core_index;
        let base_addr = self.base_addrs[core_index];
        
        // Get task parameters
        let mut task_start = job.submission.task_start;
        let mut task_number = job.submission.task_number;
        
        // Handle multi-core task distribution
        if self.config.num_irqs > 1 && !job.submission.subcore_tasks.is_empty() {
            if core_index < job.submission.subcore_tasks.len() {
                let subcore_task = &job.submission.subcore_tasks[core_index];
                task_start = subcore_task.task_start;
                task_number = subcore_task.task_number;
            }
        }
        
        let task_end = task_start + task_number - 1;
        
        // Calculate task base from memory buffer
        let task_base = job.submission.task_buffer.virt_addr.as_ptr() as *const RknpuTask;
        let first_task = unsafe { &*task_base.add(task_start as usize) };
        let last_task = unsafe { &*task_base.add(task_end as usize) };
        
        job.first_task = Some(*first_task);
        job.last_task = Some(*last_task);
        
        // Program hardware registers
        self.write_reg(base_addr, 0x10, first_task.regcmd_addr as u32)?; // PC_DATA_ADDR
        
        let pc_data_amount = (first_task.regcfg_amount + PC_DATA_EXTRA_AMOUNT + 
                             self.config.pc_data_amount_scale - 1) / self.config.pc_data_amount_scale - 1;
        self.write_reg(base_addr, 0x14, pc_data_amount)?; // PC_DATA_AMOUNT
        
        self.write_reg(base_addr, 0x20, last_task.int_mask)?; // INT_MASK
        self.write_reg(base_addr, 0x24, first_task.int_mask)?; // INT_CLEAR
        
        let task_pp_en = if job.submission.flags.ping_pong { 1 } else { 0 };
        let task_control = ((0x6 | task_pp_en) << self.config.pc_task_number_bits) | task_number;
        self.write_reg(base_addr, 0x30, task_control)?; // PC_TASK_CONTROL
        
        self.write_reg(base_addr, 0x34, job.submission.task_base_addr as u32)?; // PC_DMA_BASE_ADDR
        
        // Start operation
        self.write_reg(base_addr, 0x8, 0x1)?; // PC_OP_EN
        self.write_reg(base_addr, 0x8, 0x0)?; // PC_OP_EN
        
        job.hw_start_time = self.osal.get_time_us();
        
        self.osal.log_debug(&alloc::format!("PC job committed: start={}, num={}, addr=0x{:x}", 
                                          task_start, task_number, job.submission.task_base_addr));
        
        Ok(())
    }
    
    /// Check if job completed
    fn check_job_completion(&self, job: &HardwareJob) -> Result<bool, RknpuError> {
        let core_index = job.core_index;
        let base_addr = self.base_addrs[core_index];
        
        // Read interrupt status
        let int_status = self.read_reg(base_addr, 0x28)?; // INT_STATUS
        
        // Check if our interrupt mask matches
        if let Some(last_task) = &job.last_task {
            Ok((int_status & last_task.int_mask) != 0)
        } else {
            Ok(false)
        }
    }
    
    /// Handle interrupt for job completion
    pub fn irq_handle(&mut self, core_index: usize) -> Result<(), RknpuError> {
        if core_index >= self.base_addrs.len() {
            return Err(RknpuError::InvalidParameter);
        }
        
        let base_addr = self.base_addrs[core_index];
        
        // Read and clear interrupt status
        let int_status = self.read_reg(base_addr, 0x28)?; // INT_STATUS
        self.write_reg(base_addr, 0x24, 0x1ffff)?; // Clear all interrupts
        
        self.osal.log_debug(&alloc::format!("IRQ handled for core {}, status=0x{:x}", core_index, int_status));
        
        // Find and update corresponding job
        for job in &mut self.state.job_queue {
            if job.core_index == core_index && job.state == JobState::Running {
                if let Some(last_task) = &job.last_task {
                    if (int_status & last_task.int_mask) != 0 {
                        job.state = JobState::Completed;
                        job.hw_end_time = self.osal.get_time_us();
                        job.int_status[core_index] = int_status;
                        self.osal.log_debug(&alloc::format!("Job {} completed", job.id));
                        break;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Clear read/write amount counters
    pub fn clear_rw_amount(&self) -> Result<(), RknpuError> {
        if let Some(amount_data) = &self.config.amount_top {
            let base_addr = self.base_addrs[0];
            self.write_reg(base_addr, amount_data.offset_clr_all as u32, 1)?;
        }
        Ok(())
    }
    
    /// Get read/write amount statistics
    pub fn get_rw_amount(&self) -> Result<(u32, u32, u32), RknpuError> {
        if let Some(amount_data) = &self.config.amount_top {
            let base_addr = self.base_addrs[0];
            let dt_wr = self.read_reg(base_addr, amount_data.offset_dt_wr as u32)?;
            let dt_rd = self.read_reg(base_addr, amount_data.offset_dt_rd as u32)?;
            let wt_rd = self.read_reg(base_addr, amount_data.offset_wt_rd as u32)?;
            Ok((dt_wr, dt_rd, wt_rd))
        } else {
            Ok((0, 0, 0))
        }
    }
    
    /// Cancel all pending jobs
    fn cancel_all_jobs(&mut self) {
        for job in &mut self.state.job_queue {
            if job.state == JobState::Running || job.state == JobState::Pending {
                job.state = JobState::Error;
            }
        }
        self.osal.log_debug("All jobs cancelled");
    }
    
    /// Get core index from core mask
    fn get_core_index_from_mask(&self, core_mask: u32) -> usize {
        match core_mask {
            RKNPU_CORE0_MASK => 0,
            RKNPU_CORE1_MASK => 1,
            RKNPU_CORE2_MASK => 2,
            _ => 0, // Default to core 0
        }
    }
    
    /// Write to hardware register
    fn write_reg(&self, base_addr: NonNull<u8>, offset: u32, value: u32) -> Result<(), RknpuError> {
        unsafe {
            let reg_ptr = (base_addr.as_ptr() as *mut u32).add((offset / 4) as usize);
            core::ptr::write_volatile(reg_ptr, value);
        }
        Ok(())
    }
    
    /// Read from hardware register
    fn read_reg(&self, base_addr: NonNull<u8>, offset: u32) -> Result<u32, RknpuError> {
        unsafe {
            let reg_ptr = (base_addr.as_ptr() as *const u32).add((offset / 4) as usize);
            Ok(core::ptr::read_volatile(reg_ptr))
        }
    }
}

/// Platform-specific state initialization function type
pub type StateInitFn<O> = fn(&mut HardwareLayer<O>) -> Result<(), RknpuError>;