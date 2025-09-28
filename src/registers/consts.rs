//! Raw offsets and constants that mirror the legacy C driver macros.

/// Offset of the hardware version register.
pub const OFFSET_VERSION: usize = 0x0000;
/// Offset of the hardware version number register.
pub const OFFSET_VERSION_NUM: usize = 0x0004;
/// Offset of the program-control enable register.
pub const OFFSET_PC_OP_EN: usize = 0x0008;
/// Offset of the program-control command address register.
pub const OFFSET_PC_DATA_ADDR: usize = 0x0010;
/// Offset of the program-control command amount register.
pub const OFFSET_PC_DATA_AMOUNT: usize = 0x0014;
/// Offset of the interrupt mask register.
pub const OFFSET_INT_MASK: usize = 0x0020;
/// Offset of the interrupt clear register.
pub const OFFSET_INT_CLEAR: usize = 0x0024;
/// Offset of the interrupt status register.
pub const OFFSET_INT_STATUS: usize = 0x0028;
/// Offset of the raw interrupt status register.
pub const OFFSET_INT_RAW_STATUS: usize = 0x002C;
/// Offset of the program-control task control register.
pub const OFFSET_PC_TASK_CONTROL: usize = 0x0030;
/// Offset of the program-control DMA base address register.
pub const OFFSET_PC_DMA_BASE_ADDR: usize = 0x0034;
/// Offset of the global enable mask register.
pub const OFFSET_ENABLE_MASK: usize = 0xF008;

/// Special command offsets used on multi-core variants of the NPU.
pub const MULTICORE_COMMAND_OFFSETS: [usize; 2] = [0x1004, 0x3004];

/// Value written to acknowledge all interrupt sources.
pub const INT_CLEAR_ALL: u32 = 0x1_FFFF;

/// Additional words tagged onto the PC data payload by hardware.
pub const PC_DATA_EXTRA_AMOUNT: u32 = 4;
