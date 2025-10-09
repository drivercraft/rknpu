use ::core::ptr::NonNull;
use tock_registers::register_structs;
use tock_registers::registers::ReadWrite;

register_structs! {
    pub GlobalRegs {
        (0x0000 => _reserved0),
        (0x0008 => pub enable_mask: ReadWrite<u32, GLOBAL::Register>),
        (0x000C => @END),
    }
}

tock_registers::register_bitfields! {u32,
    GLOBAL [
        ENABLE_MASK OFFSET(0) NUMBITS(32) []
    ]
}

pub struct GlobalRegisters {
    base: NonNull<GlobalRegs>,
}

impl GlobalRegisters {
    pub const unsafe fn from_base(base: NonNull<GlobalRegs>) -> Self {
        Self { base }
    }

    fn as_ptr(&self) -> *const GlobalRegs {
        self.base.as_ptr()
    }

    pub unsafe fn regs(&self) -> &'static GlobalRegs {
        unsafe { &*self.as_ptr() }
    }

    pub unsafe fn regs_mut(&mut self) -> &'static mut GlobalRegs {
        unsafe { &mut *(self.base.as_ptr()) }
    }
}
