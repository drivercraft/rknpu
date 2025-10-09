//! Minimal GEM buffer management matching the Rockchip NPU driver layout.
//!
//! The original Linux driver relies on the DRM subsystem together with iommu
//! helpers and dma-buf bookkeeping.  In this Rust port we only keep enough
//! structure to model buffer creation, destruction and simple task storage so
//! higher level logic (such as job submission) can be validated without an
//! operating system.

#![allow(dead_code)]

use alloc::{boxed::Box, collections::btree_map::BTreeMap, vec, vec::Vec};
use core::{mem, slice};

use crate::{RknpuError, job::RknpuTask};

/// Handle identifying a GEM object inside the manager.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct RknpuGemHandle(u32);

impl RknpuGemHandle {
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    pub const fn as_raw(self) -> u64 {
        self.0 as u64
    }

    pub fn from_raw(raw: u64) -> Option<Self> {
        if raw == 0 || raw > u64::from(u32::MAX) {
            None
        } else {
            Some(Self(raw as u32))
        }
    }
}

/// Internal storage for GEM backing memory.
#[derive(Debug)]
enum GemBacking {
    Raw(Vec<u8>),
    Tasks(Vec<RknpuTask>),
}

impl GemBacking {
    fn len_bytes(&self) -> usize {
        match self {
            GemBacking::Raw(data) => data.len(),
            GemBacking::Tasks(tasks) => tasks.len() * mem::size_of::<RknpuTask>(),
        }
    }

    fn as_raw_slice(&self) -> &[u8] {
        match self {
            GemBacking::Raw(data) => data.as_slice(),
            GemBacking::Tasks(tasks) => unsafe {
                slice::from_raw_parts(
                    tasks.as_ptr() as *const u8,
                    tasks.len() * mem::size_of::<RknpuTask>(),
                )
            },
        }
    }

    fn as_raw_slice_mut(&mut self) -> &mut [u8] {
        match self {
            GemBacking::Raw(data) => data.as_mut_slice(),
            GemBacking::Tasks(tasks) => unsafe {
                slice::from_raw_parts_mut(
                    tasks.as_mut_ptr() as *mut u8,
                    tasks.len() * mem::size_of::<RknpuTask>(),
                )
            },
        }
    }

    fn as_tasks(&self) -> Option<&[RknpuTask]> {
        match self {
            GemBacking::Tasks(tasks) => Some(tasks.as_slice()),
            _ => None,
        }
    }

    fn as_tasks_mut(&mut self) -> Option<&mut [RknpuTask]> {
        match self {
            GemBacking::Tasks(tasks) => Some(tasks.as_mut_slice()),
            _ => None,
        }
    }
}

/// Simplified representation of Rockchip GEM buffer metadata.
#[derive(Debug)]
pub struct RknpuGemObject {
    handle: RknpuGemHandle,
    flags: u32,
    size: usize,
    sram_size: usize,
    iommu_domain_id: i32,
    core_mask: u32,
    backing: GemBacking,
}

impl RknpuGemObject {
    fn new(
        handle: RknpuGemHandle,
        flags: u32,
        size: usize,
        sram_size: usize,
        iommu_domain_id: i32,
        core_mask: u32,
        backing: GemBacking,
    ) -> Self {
        Self {
            handle,
            flags,
            size,
            sram_size,
            iommu_domain_id,
            core_mask,
            backing,
        }
    }

    pub fn handle(&self) -> RknpuGemHandle {
        self.handle
    }

    pub fn flags(&self) -> u32 {
        self.flags
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn iommu_domain_id(&self) -> i32 {
        self.iommu_domain_id
    }

    pub fn core_mask(&self) -> u32 {
        self.core_mask
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.backing.as_raw_slice()
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.backing.as_raw_slice_mut()
    }

    pub fn as_tasks(&self) -> Option<&[RknpuTask]> {
        self.backing.as_tasks()
    }

    pub fn as_tasks_mut(&mut self) -> Option<&mut [RknpuTask]> {
        self.backing.as_tasks_mut()
    }
}

/// Lightweight GEM object manager used by the Rust driver port.
#[derive(Default)]
pub struct RknpuGemManager {
    next_handle: u32,
    objects: BTreeMap<u32, Box<RknpuGemObject>>,
}

impl RknpuGemManager {
    pub fn new() -> Self {
        Self {
            next_handle: 1,
            objects: BTreeMap::new(),
        }
    }

    /// Allocate a zero-initialised raw buffer.
    pub fn create(
        &mut self,
        flags: u32,
        size: usize,
        sram_size: usize,
        iommu_domain_id: i32,
        core_mask: u32,
    ) -> Result<RknpuGemHandle, RknpuError> {
        if size == 0 {
            return Err(RknpuError::InvalidParameter);
        }
        let backing = GemBacking::Raw(vec![0u8; size]);
        self.insert(backing, flags, size, sram_size, iommu_domain_id, core_mask)
    }

    /// Create a task buffer from explicit task descriptors.
    pub fn create_from_tasks(
        &mut self,
        tasks: Vec<RknpuTask>,
        flags: u32,
        iommu_domain_id: i32,
        core_mask: u32,
    ) -> Result<RknpuGemHandle, RknpuError> {
        if tasks.is_empty() {
            return Err(RknpuError::InvalidParameter);
        }
        let size = tasks.len() * mem::size_of::<RknpuTask>();
        self.insert(
            GemBacking::Tasks(tasks),
            flags,
            size,
            0,
            iommu_domain_id,
            core_mask,
        )
    }

    fn insert(
        &mut self,
        backing: GemBacking,
        flags: u32,
        size: usize,
        sram_size: usize,
        iommu_domain_id: i32,
        core_mask: u32,
    ) -> Result<RknpuGemHandle, RknpuError> {
        let handle = self.allocate_handle();
        let object = RknpuGemObject::new(
            handle,
            flags,
            size,
            sram_size,
            iommu_domain_id,
            core_mask,
            backing,
        );
        self.objects.insert(handle.raw(), Box::new(object));
        Ok(handle)
    }

    fn allocate_handle(&mut self) -> RknpuGemHandle {
        // Skip the zero handle to retain parity with the C driver.
        if self.next_handle == 0 {
            self.next_handle = 1;
        }
        let handle = RknpuGemHandle::new(self.next_handle);
        self.next_handle = self.next_handle.wrapping_add(1);
        handle
    }

    pub fn destroy(&mut self, handle: RknpuGemHandle) -> Result<(), RknpuError> {
        self.objects
            .remove(&handle.raw())
            .map(|_| ())
            .ok_or(RknpuError::InvalidHandle)
    }

    pub fn get(&self, handle: RknpuGemHandle) -> Option<&RknpuGemObject> {
        self.objects.get(&handle.raw()).map(|obj| obj.as_ref())
    }

    pub fn get_mut(&mut self, handle: RknpuGemHandle) -> Option<&mut RknpuGemObject> {
        self.objects.get_mut(&handle.raw()).map(|obj| obj.as_mut())
    }

    pub fn task_slice(&self, handle: RknpuGemHandle) -> Option<&[RknpuTask]> {
        self.get(handle).and_then(|obj| obj.as_tasks())
    }

    pub fn task_slice_mut(&mut self, handle: RknpuGemHandle) -> Option<&mut [RknpuTask]> {
        self.get_mut(handle).and_then(|obj| obj.as_tasks_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_destroy_raw_buffer() {
        let mut manager = RknpuGemManager::new();
        let handle = manager
            .create(0, 128, 0, 0, 0)
            .expect("raw buffer creation should succeed");
        let obj = manager.get(handle).unwrap();
        assert_eq!(obj.size(), 128);
        assert!(obj.as_bytes().iter().all(|&byte| byte == 0));
        manager.destroy(handle).unwrap();
        assert!(manager.get(handle).is_none());
    }

    #[test]
    fn task_buffer_roundtrip() {
        let mut manager = RknpuGemManager::new();
        let tasks = vec![RknpuTask::default(); 3];
        let handle = manager
            .create_from_tasks(tasks, 0, 1, 0)
            .expect("task buffer creation should succeed");
        let slice = manager.task_slice(handle).expect("tasks slice available");
        assert_eq!(slice.len(), 3);
        manager.destroy(handle).unwrap();
    }
}
