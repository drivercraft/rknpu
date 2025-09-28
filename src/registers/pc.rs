use core::ptr::NonNull;
use tock_registers::register_structs;
use tock_registers::registers::{ReadOnly, ReadWrite};
// `register_bitfields!` macro is provided by the tock-registers crate at the
// crate root and can be invoked directly as `tock_registers::register_bitfields!`.

register_structs! {
    pub PcRegs {
        (0x0000 => pub version: ReadOnly<u32>),
        (0x0004 => pub version_num: ReadOnly<u32>),
        (0x0008 => pub pc_op_en: ReadWrite<u32>),
        (0x000C => _reserved0),
        (0x0010 => pub pc_data_addr: ReadWrite<u32>),
        (0x0014 => pub pc_data_amount: ReadWrite<u32>),
        (0x0018 => _reserved1),
        (0x0030 => pub pc_task_control: ReadWrite<u32>),
        (0x0034 => pub pc_dma_base_addr: ReadWrite<u32>),
        (0x0038 => @END),
    }
}

pub struct PcRegisters {
    base: NonNull<u8>,
}

impl PcRegisters {
    pub const unsafe fn from_base(base: NonNull<u8>) -> Self {
        Self { base }
    }

    /// Returns a pointer to the underlying register block.
    pub fn as_ptr(&self) -> *const PcRegs {
        self.base.as_ptr() as *const PcRegs
    }

    /// Returns a mutable pointer to the underlying register block.
    pub fn as_mut_ptr(&self) -> *mut PcRegs {
        self.base.as_ptr() as *mut PcRegs
    }

    /// Safety: callers must ensure exclusive access when writing.
    /// Safety: callers must ensure the MMIO mapping is valid for the lifetime
    /// of the returned reference and that concurrency/aliasing rules are
    /// respected.
    pub unsafe fn regs(&self) -> &'static PcRegs {
        // SAFETY: caller contract documented above.
        unsafe { &*self.as_ptr() }
    }

    /// Safety: callers must ensure exclusive mutable access.
    pub unsafe fn regs_mut(&self) -> &'static mut PcRegs {
        // SAFETY: caller contract documented above.
        unsafe { &mut *self.as_mut_ptr() }
    }
}

tock_registers::register_bitfields! {u32,
    PC_TASK_CONTROL [
        TASK_NUMBER OFFSET(0) NUMBITS(16) [],
        TASK_CTRL OFFSET(16) NUMBITS(16) []
    ],

    PC_OP_EN [
        ENABLE OFFSET(0) NUMBITS(1) []
    ]
}

impl PcRegisters {
    /// Build a value suitable for writing into PC_TASK_CONTROL.
    ///
    /// The driver composes the value as: ((0x6 | task_pp_en) << pc_task_number_bits) | task_number
    pub fn build_pc_task_control(
        pc_task_number_bits: u8,
        task_pp_en: bool,
        task_number: u32,
    ) -> u32 {
        let ctrl = (0x6u32 | (task_pp_en as u32)) << pc_task_number_bits;
        let mask = if pc_task_number_bits >= 32 {
            u32::MAX
        } else {
            (1u32 << pc_task_number_bits) - 1
        };
        (ctrl) | (task_number & mask)
    }

    /// Build multicore command value the driver writes to multicore offsets.
    /// Example in C: (0xe + 0x10000000 * i)
    pub fn multicore_command_value(core_index: u32) -> u32 {
        0xeu32.wrapping_add(0x10000000u32.wrapping_mul(core_index))
    }
}
