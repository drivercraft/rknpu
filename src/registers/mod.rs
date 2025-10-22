//! Memory-mapped register definitions for the Rockchip NPU.
//!
//! The register layout is described using [`tock_registers`], which provides a
//! safe and zero-cost abstraction over volatile MMIO access.  Each functional
//! block is exposed through a dedicated sub-module so the code that drives the
//! hardware can depend on a well-structured Rust API instead of scattering raw
//! offsets across the driver.

use ::core::ptr::NonNull;

pub mod cna;
pub mod consts;
pub mod core;
pub mod ddma;
pub mod dpu;
pub mod dpu_rdma;
pub mod global;
pub mod int;
pub mod pc;
pub mod ppu;
pub mod ppu_rdma;
pub mod sdma;

use consts::*;

use tock_registers::interfaces::{Readable, Writeable};

use crate::{JobMode, RknpuError, RknpuTask, data::RknpuData, registers::int::IntRegs};

const RKNPU_PC_DATA_EXTRA_AMOUNT: u32 = 4;

/// Immutable view over the RKNN register window.
pub struct RknpuCore {
    base: NonNull<u8>,
}
unsafe impl Send for RknpuCore {}

impl RknpuCore {
    /// # Safety
    ///
    /// Caller must ensure the pointer maps the RKNN register space for the
    /// lifetime of the returned structure.
    pub const unsafe fn new(base_addr: NonNull<u8>) -> Self {
        Self { base: base_addr }
    }

    #[inline(always)]
    pub(crate) fn offset_ptr<T>(&self, offset: usize) -> NonNull<T> {
        // SAFETY: caller guarantees the MMIO mapping is valid; offsets come
        // from the hardware documentation.
        unsafe {
            let ptr = self.base.as_ptr().add(offset);
            NonNull::new_unchecked(ptr as *mut T)
        }
    }

    pub fn pc(&self) -> &pc::PcRegs {
        unsafe { self.offset_ptr(PC_BASE_OFFSET).as_ref() }
    }

    pub fn int(&self) -> &IntRegs {
        unsafe { self.offset_ptr(INT_BASE_OFFSET).as_ref() }
    }

    pub fn cna(&self) -> &cna::CnaRegs {
        unsafe { self.offset_ptr(CNA_BASE_OFFSET).as_ref() }
    }

    pub fn core(&self) -> &core::CoreRegs {
        unsafe { self.offset_ptr(CORE_BASE_OFFSET).as_ref() }
    }

    pub fn dpu(&self) -> &dpu::DpuRegs {
        unsafe { self.offset_ptr(DPU_BASE_OFFSET).as_ref() }
    }

    pub fn dpu_rdma(&self) -> &dpu_rdma::DpuRdmaRegs {
        unsafe { self.offset_ptr(DPU_RDMA_BASE_OFFSET).as_ref() }
    }

    pub fn ppu(&self) -> &ppu::PpuRegs {
        unsafe { self.offset_ptr(PPU_BASE_OFFSET).as_ref() }
    }

    pub fn ppu_rdma(&self) -> &ppu_rdma::PpuRdmaRegs {
        unsafe { self.offset_ptr(PPU_RDMA_BASE_OFFSET).as_ref() }
    }

    pub fn ddma(&self) -> &ddma::DdmaRegs {
        unsafe { self.offset_ptr(DDMA_BASE_OFFSET).as_ref() }
    }

    pub fn sdma(&self) -> &sdma::SdmaRegs {
        unsafe { self.offset_ptr(SDMA_BASE_OFFSET).as_ref() }
    }

    pub fn global(&self) -> &global::GlobalRegs {
        unsafe { self.offset_ptr(GLOBAL_BASE_OFFSET).as_ref() }
    }

    pub fn version(&self) -> u32 {
        self.pc().version()
    }

    pub fn submit(
        &mut self,
        config: &RknpuData,
        flags: JobMode,
        tasks: &[RknpuTask],
        task_base_addr: u32,
        core_idx: usize,
    ) -> Result<usize, RknpuError> {
        if tasks.is_empty() {
            return Ok(0);
        }

        let pc_data_amount_scale = config.pc_data_amount_scale;

        self.pc().base_address.set(1);

        let task_pp_en = if flags.contains(JobMode::PINGPONG) {
            1
        } else {
            0
        };
        let pc_task_number_bits = config.pc_task_number_bits;

        if config.irqs.get(core_idx).is_some() {
            let val = 0xe + 0x10000000 * core_idx as u32;
            self.cna().s_pointer.set(val);
            self.core().s_pointer.set(val);
        }

        let submit_tasks = if tasks.len() > config.max_submit_number as usize {
            &tasks[0..config.max_submit_number as usize]
        } else {
            tasks
        };

        self.pc()
            .base_address
            .set(submit_tasks[0].regcmd_addr as u32);

        let amount = (submit_tasks[0].regcfg_amount + RKNPU_PC_DATA_EXTRA_AMOUNT)
            .div_ceil(pc_data_amount_scale)
            - 1;
        self.pc().register_amounts.set(amount);

        self.pc()
            .interrupt_mask
            .set(submit_tasks.last().unwrap().int_mask);
        self.pc()
            .interrupt_clear
            .set(submit_tasks.last().unwrap().int_clear);
        let task_number = submit_tasks.len() as u32;

        self.pc()
            .task_control
            .set(((0x6 | task_pp_en) << pc_task_number_bits) | task_number);

        self.pc().task_dma_base_addr.set(task_base_addr);

        self.pc().operation_enable.set(1);
        self.pc().operation_enable.set(0);

        for task in submit_tasks {
            debug!("Submitted task: {:#x?}", task);
        }

        Ok(submit_tasks.len())
    }

    pub fn handle_interrupt(&mut self) -> u32 {
        let int_status = self.pc().interrupt_status.get();

        self.pc().interrupt_clear.set(INT_CLEAR_ALL);

        rknpu_fuzz_status(int_status)
    }
}

#[inline(always)]
pub fn rknpu_fuzz_status(status: u32) -> u32 {
    let mut fuzz_status = 0;
    if (status & 0x3) != 0 {
        fuzz_status |= 0x3;
    }
    if (status & 0xc) != 0 {
        fuzz_status |= 0xc;
    }
    if (status & 0x30) != 0 {
        fuzz_status |= 0x30;
    }
    if (status & 0xc0) != 0 {
        fuzz_status |= 0xc0;
    }
    if (status & 0x300) != 0 {
        fuzz_status |= 0x300;
    }
    if (status & 0xc00) != 0 {
        fuzz_status |= 0xc00;
    }
    fuzz_status
}
