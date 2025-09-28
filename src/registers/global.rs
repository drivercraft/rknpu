use core::ptr::NonNull;
use tock_registers::register_structs;
use tock_registers::registers::ReadWrite;

register_structs! {
    pub GlobalRegs {
        (0x0000 => pub enable_mask: ReadWrite<u32, GLOBAL::Register>),
        (0x0004 => @END),
    }
}

tock_registers::register_bitfields! {u32,
    GLOBAL [
        ENABLE_MASK OFFSET(0) NUMBITS(32) []
    ]
}

pub struct GlobalRegisters {
    base: NonNull<u8>,
}

impl GlobalRegisters {
    /// Create a view for global registers. These live at a high offset
    /// (0xF000 + 0x8 -> 0xF008) from the core base; for simplicity we assume
    /// the caller will pass the core base and we compute the address.
    pub const unsafe fn from_base(base: NonNull<u8>) -> Self {
        let ptr = base.as_ptr();
        // offset of 0xF008 from core base
        let off = 0xF008usize;
        let new = unsafe { NonNull::new_unchecked(ptr.add(off)) };
        Self { base: new }
    }

    fn as_ptr(&self) -> *const GlobalRegs {
        self.base.as_ptr() as *const GlobalRegs
    }

    pub unsafe fn regs(&self) -> &'static GlobalRegs {
        unsafe { &*self.as_ptr() }
    }
}
