use crate::job::{RKNPU_MAX_SUBCORE_TASKS, RknpuSubcoreTask, RknpuSubmit};
use crate::{Rknpu, RknpuError};
/**
 * struct rknpu_subcore_task structure for subcore task index
 *
 * @task_start: task start index
 * @task_number: task number
 *
 */
struct rknpu_subcore_task {
	__u32 task_start;
	__u32 task_number;
};

/**
 * struct rknpu_submit structure for job submit
 *
 * @flags: flags for job submit
 * @timeout: submit timeout
 * @task_start: task start index
 * @task_number: task number
 * @task_counter: task counter
 * @priority: submit priority
 * @task_obj_addr: address of task object
 * @iommu_domain_id: iommu domain id
 * @reserved: just padding to be 64-bit aligned.
 * @task_base_addr: task base address
 * @hw_elapse_time: hardware elapse time
 * @core_mask: core mask of rknpu
 * @fence_fd: dma fence fd
 * @subcore_task: subcore task
 *
 */
struct rknpu_submit {
	__u32 flags;
	__u32 timeout;
	__u32 task_start;
	__u32 task_number;
	__u32 task_counter;
	__s32 priority;
	__u64 task_obj_addr;
	__u32 iommu_domain_id;
	__u32 reserved;
	__u64 task_base_addr;
	__s64 hw_elapse_time;
	__u32 core_mask;
	__s32 fence_fd;
	struct rknpu_subcore_task subcore_task[5];
};


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
        self.submit_job(args)
    }
}
