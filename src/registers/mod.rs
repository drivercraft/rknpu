//! Memory-mapped register definitions for the Rockchip NPU.
//!
//! The register layout is described using [`tock_registers`], which provides a
//! safe and zero-cost abstraction over volatile MMIO access.  Each functional
//! block is exposed through a dedicated sub-module so the code that drives the
//! hardware can depend on a well-structured Rust API instead of scattering raw
//! offsets across the driver.

use ::core::ptr::NonNull;
use tock_registers::interfaces::Readable;

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

/// Immutable view over the RKNN register window.
pub struct RknpuRegisters {
    base: NonNull<u8>,
}
unsafe impl Send for RknpuRegisters {}

impl RknpuRegisters {
    /// # Safety
    ///
    /// Caller must ensure the pointer maps the RKNN register space for the
    /// lifetime of the returned structure.
    pub const unsafe fn new(base_addr: NonNull<u8>) -> Self {
        Self { base: base_addr }
    }

    #[inline(always)]
    fn offset_ptr<T>(&self, offset: usize) -> NonNull<T> {
        // SAFETY: caller guarantees the MMIO mapping is valid; offsets come
        // from the hardware documentation.
        unsafe {
            let ptr = self.base.as_ptr().add(offset);
            NonNull::new_unchecked(ptr as *mut T)
        }
    }

    pub fn pc(&self) -> pc::PcRegisters {
        unsafe { pc::PcRegisters::from_base(self.offset_ptr(PC_BASE_OFFSET)) }
    }

    pub fn int(&self) -> int::IntRegisters {
        unsafe { int::IntRegisters::from_base(self.offset_ptr(INT_BASE_OFFSET)) }
    }

    pub fn cna(&self) -> cna::CnaRegisters {
        unsafe { cna::CnaRegisters::from_base(self.offset_ptr(CNA_BASE_OFFSET)) }
    }

    pub fn core(&self) -> core::CoreRegisters {
        unsafe { core::CoreRegisters::from_base(self.offset_ptr(CORE_BASE_OFFSET)) }
    }

    pub fn dpu(&self) -> dpu::DpuRegisters {
        unsafe { dpu::DpuRegisters::from_base(self.offset_ptr(DPU_BASE_OFFSET)) }
    }

    pub fn dpu_rdma(&self) -> dpu_rdma::DpuRdmaRegisters {
        unsafe { dpu_rdma::DpuRdmaRegisters::from_base(self.offset_ptr(DPU_RDMA_BASE_OFFSET)) }
    }

    pub fn ppu(&self) -> ppu::PpuRegisters {
        unsafe { ppu::PpuRegisters::from_base(self.offset_ptr(PPU_BASE_OFFSET)) }
    }

    pub fn ppu_rdma(&self) -> ppu_rdma::PpuRdmaRegisters {
        unsafe { ppu_rdma::PpuRdmaRegisters::from_base(self.offset_ptr(PPU_RDMA_BASE_OFFSET)) }
    }

    pub fn ddma(&self) -> ddma::DdmaRegisters {
        unsafe { ddma::DdmaRegisters::from_base(self.offset_ptr(DDMA_BASE_OFFSET)) }
    }

    pub fn sdma(&self) -> sdma::SdmaRegisters {
        unsafe { sdma::SdmaRegisters::from_base(self.offset_ptr(SDMA_BASE_OFFSET)) }
    }

    pub fn global(&self) -> global::GlobalRegisters {
        unsafe { global::GlobalRegisters::from_base(self.offset_ptr(GLOBAL_BASE_OFFSET)) }
    }

    pub fn version(&self) -> u32 {
        let pc = self.pc();
        let regs = pc.regs();
        regs.version
            .get()
            .wrapping_add(regs.version_num.get() & 0xffff)
    }
}
