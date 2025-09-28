//! RKNPU configuration bindings translated from the C `struct rknpu_config`.
//!
//! This module provides a `#[repr(C)]` Rust equivalent suitable for FFI
//! or direct translation of kernel-style configuration data.

/// Returns a mask with the lowest `n` bits set.
/// Mirrors the C macro: (((n) == 64) ? ~0ULL : ((1ULL<<(n))-1))
pub const fn dma_bit_mask(n: u32) -> u64 {
    if n >= 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1u64
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct RknpuAmountData {
    pub offset_clr_all: u16,
    pub offset_dt_wr: u16,
    pub offset_dt_rd: u16,
    pub offset_wt_rd: u16,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RknpuType {
    Rk3588,
}

#[derive(Debug, Clone)]
pub struct RknpuConfig {
    pub ty: RknpuType,
    pub bw_priority_addr: u32,
    pub bw_priority_length: u32,
    pub dma_mask: u64,
    pub pc_data_amount_scale: u32,
    pub pc_task_number_bits: u32,
    pub pc_task_number_mask: u32,
    pub pc_task_status_offset: u32,
    pub pc_dma_ctrl: u32,
    pub nbuf_phyaddr: u64,
    pub nbuf_size: u64,
    pub max_submit_number: u64,
    pub core_mask: u32,
    /// Pointer to top-level amount data (opaque).
    pub amount_top: Option<RknpuAmountData>,
    /// Pointer to per-core amount data (opaque).
    pub amount_core: Option<RknpuAmountData>,
}

impl RknpuConfig {
    pub fn new(ty: RknpuType) -> Self {
        match ty {
            RknpuType::Rk3588 => Self::new_3588(),
        }
    }

    fn new_3588() -> Self {
        Self {
            ty: RknpuType::Rk3588,
            bw_priority_addr: 0xFF5A_0000,
            bw_priority_length: 0x1000,
            dma_mask: dma_bit_mask(40),
            pc_data_amount_scale: 2,
            pc_task_number_bits: 12,
            pc_task_number_mask: 0xfff,
            pc_task_status_offset: 0x3c,
            pc_dma_ctrl: 0,
            nbuf_phyaddr: 0xFF7E_0000,
            nbuf_size: 0x20000,
            max_submit_number: (1u64 << 12) - 1u64,
            core_mask: 0x7,
            amount_top: None,
            amount_core: None,
        }
    }
}
