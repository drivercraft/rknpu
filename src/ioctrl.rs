use crate::{Rknpu, RknpuError, SubmitBase, SubmitRef};

/// 子核心任务索引结构体
///
/// 对应 C 结构体 `rknpu_subcore_task`
/// 用于表示子核心任务的起始索引和任务数量
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RknpuSubcoreTask {
    /// 任务起始索引
    pub task_start: u32,
    /// 任务数量
    pub task_number: u32,
}

/// 任务提交结构体
///
/// 对应 C 结构体 `rknpu_submit`
/// 用于向 RKNPU 提交作业任务
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RknpuSubmit {
    /// 作业提交标志
    pub flags: u32,
    /// 提交超时时间
    pub timeout: u32,
    /// 任务起始索引
    pub task_start: u32,
    /// 任务数量
    pub task_number: u32,
    /// 任务计数器
    pub task_counter: u32,
    /// 提交优先级
    pub priority: i32,
    /// 任务对象地址
    pub task_obj_addr: u64,
    /// IOMMU 域 ID
    pub iommu_domain_id: u32,
    /// 保留字段（64位对齐）
    pub reserved: u32,
    /// 任务基地址
    pub task_base_addr: u64,
    /// 硬件运行时间
    pub hw_elapse_time: i64,
    /// RKNPU 核心掩码
    pub core_mask: u32,
    /// DMA 信号量文件描述符
    pub fence_fd: i32,
    /// 子核心任务数组（固定大小为5）
    pub subcore_task: [RknpuSubcoreTask; 5],
}

/// User-desired buffer creation information structure.
///
/// Fields correspond to the original C layout. Use `#[repr(C)]` so this type
/// can be used across the FFI boundary or when mirroring kernel structs.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RknpuMemCreate {
    /// The handle of the created GEM object.
    pub handle: u32,
    /// User request for setting memory type or cache attributes.
    pub flags: u32,
    /// User-desired memory allocation size (page-aligned by caller).
    pub size: u64,
    /// Address of RKNPU memory object.
    pub obj_addr: u64,
    /// DMA address that is accessible by the RKNPU.
    pub dma_addr: u64,
    /// User-desired SRAM memory allocation size (page-aligned by caller).
    pub sram_size: u64,
    /// IOMMU domain id.
    pub iommu_domain_id: i32,
    /// Core mask (reserved/padding in original structure).
    pub core_mask: u32,
}

/// For synchronizing DMA buffer
///
/// Fields correspond to the original C layout. Use `#[repr(C)]` so this type
/// can be used across FFI boundaries if needed.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RknpuMemSync {
    /// User request for setting memory type or cache attributes.
    pub flags: u32,
    /// Reserved for padding.
    pub reserved: u32,
    /// Address of RKNPU memory object.
    pub obj_addr: u64,
    /// Offset in bytes from start address of buffer.
    pub offset: u64,
    /// Size of memory region.
    pub size: u64,
}

impl Rknpu {
    pub fn submit_ioctrl(&self, args: &mut RknpuSubmit) -> Result<(), RknpuError> {
        let mut n = 0;
        let all = args.task_number as usize;
        let max_submit_number = self.data.max_submit_number as usize;
        let mut task_start = args.task_start;
        let mut int_mask = 0;

        while n < all {
            let task_number = core::cmp::min(all - n, max_submit_number);
            task_start = args.task_start + n as u32;
            let task_end = task_start + task_number as u32 - 1;

            let job = SubmitRef {
                base: SubmitBase {
                    flags: todo!(),
                    task_base_addr: todo!(),
                    core_idx: todo!(),
                    int_mask: todo!(),
                    int_clear: todo!(),
                    regcfg_amount: todo!(),
                },
                task_number,
                regcmd_base_addr: todo!(),
            };
            self.base[0].submit_pc(&self.data, &job).unwrap();
            debug!(
                "Submitted tasks from {} to {}",
                task_start,
                task_start + task_number as u32 - 1
            );
            loop {
                let status = self.base[0].handle_interrupt();
                if status == int_mask {
                    break;
                }
            }

            debug!(
                "Completed tasks from {} to {}",
                task_start,
                task_start + task_number as u32 - 1
            );
            n += task_number;
        }

        Ok(())
    }
}
