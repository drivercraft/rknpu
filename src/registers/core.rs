use ::core::ptr::NonNull;
use tock_registers::{
    register_structs,
    registers::{ReadOnly, ReadWrite},
};

register_structs! {
    #[allow(non_snake_case)]
    pub CoreRegs {
        (0x0000 => pub s_status: ReadOnly<u32>),
        (0x0004 => pub s_pointer: ReadWrite<u32>),
        (0x0008 => pub operation_enable: ReadWrite<u32>),
        (0x000C => pub mac_gating: ReadWrite<u32>),
        (0x0010 => pub misc_cfg: ReadWrite<u32>),
        (0x0014 => pub dataout_size0: ReadWrite<u32>),
        (0x0018 => pub dataout_size1: ReadWrite<u32>),
        (0x001C => pub clip_truncate: ReadWrite<u32>),
        (0x0020 => @END),
    }
}

pub struct CoreRegisters {
    base: NonNull<CoreRegs>,
}

impl CoreRegisters {
    pub const unsafe fn from_base(base: NonNull<CoreRegs>) -> Self {
        Self { base }
    }

    #[inline]
    pub fn regs(&self) -> &CoreRegs {
        unsafe { self.base.as_ref() }
    }

    #[inline]
    pub fn regs_mut(&mut self) -> &mut CoreRegs {
        unsafe { self.base.as_mut() }
    }
}
