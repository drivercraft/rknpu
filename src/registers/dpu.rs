use ::core::ptr::NonNull;
use tock_registers::{
    register_structs,
    registers::{ReadOnly, ReadWrite},
};

register_structs! {
    #[allow(non_snake_case)]
    pub DpuRegs {
        (0x0000 => pub s_status: ReadOnly<u32>),
        (0x0004 => pub s_pointer: ReadWrite<u32>),
        (0x0008 => pub operation_enable: ReadWrite<u32>),
        (0x000C => pub feature_mode_cfg: ReadWrite<u32>),
        (0x0010 => pub data_format: ReadWrite<u32>),
        (0x0014 => pub offset_pend: ReadWrite<u32>),
        (0x0018 => _reserved0),
        (0x0020 => pub dst_base_addr: ReadWrite<u32>),
        (0x0024 => pub dst_surf_stride: ReadWrite<u32>),
        (0x0028 => _reserved1),
        (0x0030 => pub data_cube_width: ReadWrite<u32>),
        (0x0034 => pub data_cube_height: ReadWrite<u32>),
        (0x0038 => pub data_cube_notch_addr: ReadWrite<u32>),
        (0x003C => pub data_cube_channel: ReadWrite<u32>),
        (0x0040 => pub bs_cfg: ReadWrite<u32>),
        (0x0044 => pub bs_alu_cfg: ReadWrite<u32>),
        (0x0048 => pub bs_mul_cfg: ReadWrite<u32>),
        (0x004C => pub bs_relux_cmp_value: ReadWrite<u32>),
        (0x0050 => pub bs_ow_cfg: ReadWrite<u32>),
        (0x0054 => pub bs_ow_op: ReadWrite<u32>),
        (0x0058 => pub wdma_size0: ReadWrite<u32>),
        (0x005C => pub wdma_size1: ReadWrite<u32>),
        (0x0060 => pub bn_cfg: ReadWrite<u32>),
        (0x0064 => pub bn_alu_cfg: ReadWrite<u32>),
        (0x0068 => pub bn_mul_cfg: ReadWrite<u32>),
        (0x006C => pub bn_relux_cmp_value: ReadWrite<u32>),
        (0x0070 => pub ew_cfg: ReadWrite<u32>),
        (0x0074 => pub ew_cvt_offset_value: ReadWrite<u32>),
        (0x0078 => pub ew_cvt_scale_value: ReadWrite<u32>),
        (0x007C => pub ew_relux_cmp_value: ReadWrite<u32>),
        (0x0080 => pub out_cvt_offset: ReadWrite<u32>),
        (0x0084 => pub out_cvt_scale: ReadWrite<u32>),
        (0x0088 => pub out_cvt_shift: ReadWrite<u32>),
        (0x008C => _reserved2),
        (0x0090 => pub ew_op_value0: ReadWrite<u32>),
        (0x0094 => pub ew_op_value1: ReadWrite<u32>),
        (0x0098 => pub ew_op_value2: ReadWrite<u32>),
        (0x009C => pub ew_op_value3: ReadWrite<u32>),
        (0x00A0 => pub ew_op_value4: ReadWrite<u32>),
        (0x00A4 => pub ew_op_value5: ReadWrite<u32>),
        (0x00A8 => pub ew_op_value6: ReadWrite<u32>),
        (0x00AC => pub ew_op_value7: ReadWrite<u32>),
        (0x00B0 => _reserved3),
        (0x00C0 => pub surface_add: ReadWrite<u32>),
        (0x00C4 => _reserved4),
        (0x0100 => pub lut_access_cfg: ReadWrite<u32>),
        (0x0104 => pub lut_access_data: ReadWrite<u32>),
        (0x0108 => pub lut_cfg: ReadWrite<u32>),
        (0x010C => pub lut_info: ReadWrite<u32>),
        (0x0110 => pub lut_le_start: ReadWrite<u32>),
        (0x0114 => pub lut_le_end: ReadWrite<u32>),
        (0x0118 => pub lut_lo_start: ReadWrite<u32>),
        (0x011C => pub lut_lo_end: ReadWrite<u32>),
        (0x0120 => pub lut_le_slope_scale: ReadWrite<u32>),
        (0x0124 => pub lut_le_slope_shift: ReadWrite<u32>),
        (0x0128 => pub lut_lo_slope_scale: ReadWrite<u32>),
        (0x012C => pub lut_lo_slope_shift: ReadWrite<u32>),
        (0x0130 => @END),
    }
}

pub struct DpuRegisters {
    base: NonNull<DpuRegs>,
}

impl DpuRegisters {
    pub const unsafe fn from_base(base: NonNull<DpuRegs>) -> Self {
        Self { base }
    }

    #[inline]
    pub fn regs(&self) -> &DpuRegs {
        unsafe { self.base.as_ref() }
    }

    #[inline]
    pub fn regs_mut(&mut self) -> &mut DpuRegs {
        unsafe { self.base.as_mut() }
    }
}
