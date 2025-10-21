//! Minimal job submission support translated from the C driver.
//!
//! The original C implementation wires into the Linux kernel's scheduling,
//! DMA, fence, and waitqueue infrastructure.  In this Rust port we keep the
//! data layout and validation logic but replace operating system specific
//! interactions with lightweight placeholders so the higher level driver code
//! can be compiled and exercised in a freestanding environment.

#![allow(dead_code)]

use core::{fmt, sync::atomic::AtomicU32};

use dma_api::DVec;

/// Maximum number of hardware cores supported by the IP.
pub const RKNPU_MAX_CORES: usize = 3;

/// Maximum number of sub-core task descriptors accepted per submit.
pub const RKNPU_MAX_SUBCORE_TASKS: usize = 5;

/// Automatic core selection requested by the caller.
pub const RKNPU_CORE_AUTO_MASK: u32 = 0x00;
/// Explicit mask targeting core 0.
pub const RKNPU_CORE0_MASK: u32 = 0x01;
/// Explicit mask targeting core 1.
pub const RKNPU_CORE1_MASK: u32 = 0x02;
/// Explicit mask targeting core 2.
pub const RKNPU_CORE2_MASK: u32 = 0x04;

/// Job flag requesting PC (Program Counter) mode.
pub const RKNPU_JOB_PC: u32 = 1 << 0;
/// Job flag requesting non-blocking submission.
pub const RKNPU_JOB_NONBLOCK: u32 = 1 << 1;
/// Job flag enabling ping-pong execution.
pub const RKNPU_JOB_PINGPONG: u32 = 1 << 2;
/// Job flag indicating a fence should be waited on before execution.
pub const RKNPU_JOB_FENCE_IN: u32 = 1 << 3;
/// Job flag indicating a fence should be signalled on completion.
pub const RKNPU_JOB_FENCE_OUT: u32 = 1 << 4;

/// Task descriptor consumed by the hardware command parser in PC mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C, packed)]
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

/// High level view of a sub-core task request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RknpuSubcoreTask {
    pub task_start: u32,
    pub task_number: u32,
}

/// Submission descriptor mirroring the userspace ABI.
pub struct RknpuSubmit {
    pub flags: u32,
    pub timeout: u32,
    pub task_start: u32,
    pub task_number: u32,
    pub task_counter: u32,
    pub priority: i32,
    pub task_obj: DVec<u8>,
    // pub task_obj_addr: u64,
    pub iommu_domain_id: u32,
    pub reserved: u32,
    pub task_base_addr: u64,
    pub hw_elapse_time: i64,
    pub core_mask: u32,
    pub fence_fd: i32,
    pub subcore_task: [RknpuSubcoreTask; RKNPU_MAX_SUBCORE_TASKS],
}

impl fmt::Debug for RknpuSubmit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RknpuSubmit")
            .field("flags", &self.flags)
            .field("timeout", &self.timeout)
            .field("task_start", &self.task_start)
            .field("task_number", &self.task_number)
            .field("task_counter", &self.task_counter)
            .field("priority", &self.priority)
            .field("task_obj_paddr", &self.task_obj.bus_addr())
            .field("iommu_domain_id", &self.iommu_domain_id)
            .field("task_base_addr", &self.task_base_addr)
            .field("hw_elapse_time", &self.hw_elapse_time)
            .field("core_mask", &format_args!("0x{:x}", self.core_mask))
            .field("fence_fd", &self.fence_fd)
            .finish()
    }
}

impl RknpuSubmit {
    /// Returns the number of cores explicitly requested by this submission.
    pub fn requested_cores(&self) -> u32 {
        match self.core_mask {
            RKNPU_CORE_AUTO_MASK => 0,
            mask => mask.count_ones(),
        }
    }

    /// Returns true if the submission requests non-blocking execution.
    pub fn is_nonblocking(&self) -> bool {
        self.flags & RKNPU_JOB_NONBLOCK != 0
    }

    /// Returns true if PC mode was requested.
    pub fn is_pc_mode(&self) -> bool {
        self.flags & RKNPU_JOB_PC != 0
    }
}

/// Helper calculating the mask for the given core index.
pub const fn core_mask_from_index(index: usize) -> u32 {
    match index {
        0 => RKNPU_CORE0_MASK,
        1 => RKNPU_CORE1_MASK,
        2 => RKNPU_CORE2_MASK,
        _ => 0,
    }
}

/// Counts how many cores are enabled in the provided mask.
pub const fn core_count_from_mask(mask: u32) -> u32 {
    mask.count_ones()
}

#[derive(Debug)]
pub struct RknpuJob {
    /// Number of cores to use for this job.
    pub use_core_num: usize,
    pub args: RknpuSubmit,
    pub first_task: usize,
    pub last_task: usize,
    pub int_mask: [u32; RKNPU_MAX_CORES],
    pub int_status: [u32; RKNPU_MAX_CORES],
    pub submit_count: [AtomicU32; RKNPU_MAX_CORES],
}
