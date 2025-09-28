use core::ptr::NonNull;
use tock_registers::interfaces::{Readable, Writeable};
use tock_registers::register_structs;
use tock_registers::registers::{ReadOnly, ReadWrite, WriteOnly};

register_structs! {
    /// Int register block starting at offset 0 relative to the int group base.
    pub IntRegs {
        (0x00 => pub int_mask: ReadWrite<u32>),
        (0x04 => pub int_clear: WriteOnly<u32>),
        (0x08 => pub int_status: ReadOnly<u32>),
        (0x0C => pub int_raw_status: ReadOnly<u32>),
        (0x10 => @END),
    }
}

pub struct IntRegisters {
    base: NonNull<u8>,
}

impl IntRegisters {
    /// Create an IntRegisters view for the interrupt group. The provided base
    /// should be the core base; this function will offset to the interrupt
    /// register region.
    pub const unsafe fn from_base(base: NonNull<u8>) -> Self {
        // INT registers live at +0x20 from the core base.
        let ptr = base.as_ptr();
        let off = 0x20usize;
        let new = unsafe { NonNull::new_unchecked(ptr.add(off)) };
        Self { base: new }
    }

    fn as_ptr(&self) -> *const IntRegs {
        self.base.as_ptr() as *const IntRegs
    }

    fn as_mut_ptr(&self) -> *mut IntRegs {
        self.base.as_ptr() as *mut IntRegs
    }

    /// Safety: see docs on `from_base` - caller must ensure mapping validity.
    pub unsafe fn regs(&self) -> &'static IntRegs {
        unsafe { &*self.as_ptr() }
    }

    /// Safety: caller must ensure exclusive mutable access.
    pub unsafe fn regs_mut(&self) -> &'static mut IntRegs {
        unsafe { &mut *self.as_mut_ptr() }
    }
}

tock_registers::register_bitfields! {u32,
    INT_REGS [
        IRQ0 OFFSET(0) NUMBITS(1) [],
        IRQ1 OFFSET(1) NUMBITS(1) [],
        IRQ2 OFFSET(2) NUMBITS(1) [],
        IRQ_ALL OFFSET(0) NUMBITS(32) []
    ]
}

/// Value used by the Linux driver to clear all interrupt sources.
pub const INT_CLEAR_ALL: u32 = 0x1_FFFF;

impl IntRegisters {
    /// Clear all interrupts by writing the driver's `RKNPU_INT_CLEAR` value.
    ///
    /// Safety: caller must ensure MMIO mapping validity.
    pub unsafe fn clear_all(&self) {
        let regs = unsafe { self.regs_mut() };
        regs.int_clear.set(INT_CLEAR_ALL);
    }

    /// Write an interrupt mask value.
    pub unsafe fn set_mask(&self, mask: u32) {
        let regs = unsafe { self.regs_mut() };
        regs.int_mask.set(mask);
    }

    /// Read the masked interrupt status.
    pub unsafe fn read_status(&self) -> u32 {
        let regs = unsafe { self.regs() };
        regs.int_status.get()
    }

    /// Read the raw interrupt status.
    pub unsafe fn read_raw_status(&self) -> u32 {
        let regs = unsafe { self.regs() };
        regs.int_raw_status.get()
    }
}
