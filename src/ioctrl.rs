use crate::{JobMode, Rknpu, RknpuError, RknpuTask, SubmitBase, SubmitRef};

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
    pub fn submit_ioctrl(&mut self, args: &mut RknpuSubmit) -> Result<(), RknpuError> {
        self.gem.comfirm_write_all()?;
        let mut tasks = unsafe {
            core::slice::from_raw_parts(
                args.task_obj_addr as *const RknpuTask,
                args.task_number as usize,
            )
        };
        let max_submit_number = self.data.max_submit_number as usize;

        while !tasks.is_empty() {
            let submit_tasks = if tasks.len() > max_submit_number {
                &tasks[..max_submit_number]
            } else {
                tasks
            };

            let job = SubmitRef {
                base: SubmitBase {
                    flags: JobMode::from_bits_retain(args.flags),
                    task_base_addr: args.task_base_addr as _,
                    core_idx: args.core_mask.trailing_zeros() as usize,
                    int_mask: submit_tasks.last().unwrap().int_mask,
                    int_clear: submit_tasks.last().unwrap().int_clear,
                    regcfg_amount: submit_tasks[0].regcfg_amount,
                },
                task_number: submit_tasks.len(),
                regcmd_base_addr: submit_tasks[0].regcmd_addr as _,
            };
            debug!("Submit job: {job:#x?}");
            let pre_status = self.base[0].handle_interrupt();
            self.base[0].submit_pc(&self.data, &job).unwrap();

            // Wait for completion
            loop {
                let status = self.base[0].handle_interrupt();
                if status == job.base.int_mask {
                    break;
                }
                if status != pre_status {
                    debug!("Interrupt status changed: {:#x}", status);
                    return Err(RknpuError::TaskError);
                }
            }
            debug!("Job completed");
            tasks = &tasks[submit_tasks.len()..];
        }
        self.gem.prepare_read_all()?;

        Ok(())
    }
}
