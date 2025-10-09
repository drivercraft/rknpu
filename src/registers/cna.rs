use ::core::ptr::NonNull;
use tock_registers::{register_structs, registers::{ReadOnly, ReadWrite}};

register_structs! {
    #[allow(non_snake_case)]
    pub CnaRegs {
        (0x0000 => pub s_status: ReadOnly<u32>),
        (0x0004 => pub s_pointer: ReadWrite<u32>),
        (0x0008 => pub operation_enable: ReadWrite<u32>),
        (0x000C => pub conv_con1: ReadWrite<u32>),
        (0x0010 => pub conv_con2: ReadWrite<u32>),
        (0x0014 => pub conv_con3: ReadWrite<u32>),
        (0x0018 => _reserved0),
        (0x0020 => pub data_size0: ReadWrite<u32>),
        (0x0024 => pub data_size1: ReadWrite<u32>),
        (0x0028 => pub data_size2: ReadWrite<u32>),
        (0x002C => pub data_size3: ReadWrite<u32>),
        (0x0030 => pub weight_size0: ReadWrite<u32>),
        (0x0034 => pub weight_size1: ReadWrite<u32>),
        (0x0038 => pub weight_size2: ReadWrite<u32>),
        (0x003C => _reserved1),
        (0x0040 => pub cbuf_con0: ReadWrite<u32>),
        (0x0044 => pub cbuf_con1: ReadWrite<u32>),
        (0x0048 => _reserved2),
        (0x004C => pub cvt_con0: ReadWrite<u32>),
        (0x0050 => pub cvt_con1: ReadWrite<u32>),
        (0x0054 => pub cvt_con2: ReadWrite<u32>),
        (0x0058 => pub cvt_con3: ReadWrite<u32>),
        (0x005C => pub cvt_con4: ReadWrite<u32>),
        (0x0060 => pub fc_con0: ReadWrite<u32>),
        (0x0064 => pub fc_con1: ReadWrite<u32>),
        (0x0068 => pub pad_con0: ReadWrite<u32>),
        (0x006C => pub feature_data_addr: ReadWrite<u32>),
        (0x0070 => pub fc_con2: ReadWrite<u32>),
        (0x0074 => pub dma_con0: ReadWrite<u32>),
        (0x0078 => pub dma_con1: ReadWrite<u32>),
        (0x007C => pub dma_con2: ReadWrite<u32>),
        (0x0080 => pub fc_data_size0: ReadWrite<u32>),
        (0x0084 => pub fc_data_size1: ReadWrite<u32>),
        (0x0088 => _reserved3),
        (0x0090 => pub clk_gate: ReadWrite<u32>),
        (0x0094 => _reserved4),
        (0x0100 => pub dcomp_ctrl: ReadWrite<u32>),
        (0x0104 => pub dcomp_regnum: ReadWrite<u32>),
        (0x0108 => _reserved5),
        (0x0110 => pub dcomp_addr0: ReadWrite<u32>),
        (0x0114 => _reserved6),
        (0x0140 => pub dcomp_amount0: ReadWrite<u32>),
        (0x0144 => pub dcomp_amount1: ReadWrite<u32>),
        (0x0148 => pub dcomp_amount2: ReadWrite<u32>),
        (0x014C => pub dcomp_amount3: ReadWrite<u32>),
        (0x0150 => pub dcomp_amount4: ReadWrite<u32>),
        (0x0154 => pub dcomp_amount5: ReadWrite<u32>),
        (0x0158 => pub dcomp_amount6: ReadWrite<u32>),
        (0x015C => pub dcomp_amount7: ReadWrite<u32>),
        (0x0160 => pub dcomp_amount8: ReadWrite<u32>),
        (0x0164 => pub dcomp_amount9: ReadWrite<u32>),
        (0x0168 => pub dcomp_amount10: ReadWrite<u32>),
        (0x016C => pub dcomp_amount11: ReadWrite<u32>),
        (0x0170 => pub dcomp_amount12: ReadWrite<u32>),
        (0x0174 => pub dcomp_amount13: ReadWrite<u32>),
        (0x0178 => pub dcomp_amount14: ReadWrite<u32>),
        (0x017C => pub dcomp_amount15: ReadWrite<u32>),
        (0x0180 => pub cvt_con5: ReadWrite<u32>),
        (0x0184 => pub pad_con1: ReadWrite<u32>),
        (0x0188 => @END),
    }
}

pub struct CnaRegisters {
    base: NonNull<CnaRegs>,
}

impl CnaRegisters {
    pub const unsafe fn from_base(base: NonNull<CnaRegs>) -> Self {
        Self { base }
    }

    #[inline]
    pub fn regs(&self) -> &CnaRegs {
        unsafe { self.base.as_ref() }
    }

    #[inline]
    pub fn regs_mut(&mut self) -> &mut CnaRegs {
        unsafe { self.base.as_mut() }
    }
}
