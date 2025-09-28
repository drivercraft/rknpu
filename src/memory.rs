//! Memory management for RKNPU device layer
//! 
//! This module provides memory allocation and management functions
//! for NPU operations including DMA buffers, task buffers, and SRAM management.

use core::ptr::NonNull;
use alloc::vec::Vec;
use crate::osal::{Osal, OsalError, MemoryBuffer, MemoryFlags, DmaSyncDirection, SramManager, NBufMapping, ScatterGatherTable};
use crate::err::RknpuError;

/// Memory object handle
pub type MemHandle = u32;

/// Memory allocation flags specific to NPU
#[derive(Debug, Clone, Copy)]
pub struct NpuMemoryFlags {
    pub base_flags: MemoryFlags,
    pub iommu: bool,
    pub sram: bool,
    pub nbuf: bool,
    pub secure: bool,
    pub kernel_mapping: bool,
    pub iova_alignment: bool,
}

impl Default for NpuMemoryFlags {
    fn default() -> Self {
        Self {
            base_flags: MemoryFlags::default(),
            iommu: false,
            sram: false,
            nbuf: false,
            secure: false,
            kernel_mapping: false,
            iova_alignment: false,
        }
    }
}

/// Memory object descriptor
#[derive(Debug)]
pub struct MemoryObject {
    pub handle: MemHandle,
    pub buffer: MemoryBuffer,
    pub flags: NpuMemoryFlags,
    pub ref_count: u32,
    pub sgt: Option<ScatterGatherTable>,
}

impl MemoryObject {
    pub fn new(handle: MemHandle, buffer: MemoryBuffer, flags: NpuMemoryFlags) -> Self {
        Self {
            handle,
            buffer,
            flags,
            ref_count: 1,
            sgt: None,
        }
    }
    
    pub fn add_ref(&mut self) {
        self.ref_count += 1;
    }
    
    pub fn release(&mut self) -> bool {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
        self.ref_count == 0
    }
}

/// Memory manager for NPU device
pub struct MemoryManager<O: Osal> {
    osal: O,
    objects: Vec<MemoryObject>,
    next_handle: MemHandle,
    sram_manager: Option<SramManager>,
    nbuf_mapping: Option<NBufMapping>,
    iommu_enabled: bool,
}

impl<O: Osal> MemoryManager<O> {
    /// Create new memory manager
    pub fn new(osal: O, iommu_enabled: bool) -> Self {
        Self {
            osal,
            objects: Vec::new(),
            next_handle: 1,
            sram_manager: None,
            nbuf_mapping: None,
            iommu_enabled,
        }
    }
    
    /// Initialize SRAM manager
    pub fn init_sram(&mut self, start_addr: u64, size: usize, chunk_size: usize) -> Result<(), RknpuError> {
        if size == 0 || chunk_size == 0 {
            return Err(RknpuError::InvalidParameter);
        }
        
        self.sram_manager = Some(SramManager::new(start_addr, size, chunk_size));
        self.osal.log_info(&alloc::format!("SRAM initialized: start=0x{:x}, size=0x{:x}, chunk_size=0x{:x}",
                                          start_addr, size, chunk_size));
        Ok(())
    }
    
    /// Initialize NBUF mapping
    pub fn init_nbuf(&mut self, phys_start: u64, size: usize) -> Result<(), RknpuError> {
        if size == 0 {
            return Err(RknpuError::InvalidParameter);
        }
        
        self.nbuf_mapping = Some(NBufMapping::new(phys_start, size));
        self.osal.log_info(&alloc::format!("NBUF initialized: start=0x{:x}, size=0x{:x}",
                                          phys_start, size));
        Ok(())
    }
    
    /// Allocate memory object
    pub fn allocate(&mut self, size: usize, flags: NpuMemoryFlags) -> Result<MemHandle, RknpuError> {
        if size == 0 {
            return Err(RknpuError::InvalidParameter);
        }
        
        let handle = self.next_handle;
        self.next_handle += 1;
        
        // Try SRAM allocation if requested
        if flags.sram {
            if let Some(ref mut sram_mgr) = self.sram_manager {
                let chunks_needed = (size + sram_mgr.chunk_size - 1) / sram_mgr.chunk_size;
                if let Some(sram_addr) = sram_mgr.allocate_chunks(chunks_needed) {
                    // Create SRAM memory object (simplified)
                    let buffer = MemoryBuffer {
                        virt_addr: NonNull::new(sram_addr as *mut u8).ok_or(RknpuError::InvalidParameter)?,
                        phys_addr: sram_addr,
                        dma_addr: sram_addr,
                        size: chunks_needed * sram_mgr.chunk_size,
                        flags: flags.base_flags,
                    };
                    
                    let mem_obj = MemoryObject::new(handle, buffer, flags);
                    self.objects.push(mem_obj);
                    
                    self.osal.log_debug(&alloc::format!("SRAM allocated: handle={}, addr=0x{:x}, size=0x{:x}",
                                                       handle, sram_addr, size));
                    return Ok(handle);
                }
            }
        }
        
        // Try NBUF allocation if requested
        if flags.nbuf {
            if let Some(ref nbuf) = self.nbuf_mapping {
                if size <= nbuf.size {
                    // Simplified NBUF allocation - in practice would need proper management
                    let buffer = MemoryBuffer {
                        virt_addr: NonNull::new(nbuf.phys_start as *mut u8).ok_or(RknpuError::InvalidParameter)?,
                        phys_addr: nbuf.phys_start,
                        dma_addr: nbuf.phys_start,
                        size,
                        flags: flags.base_flags,
                    };
                    
                    let mem_obj = MemoryObject::new(handle, buffer, flags);
                    self.objects.push(mem_obj);
                    
                    self.osal.log_debug(&alloc::format!("NBUF allocated: handle={}, addr=0x{:x}, size=0x{:x}",
                                                       handle, nbuf.phys_start, size));
                    return Ok(handle);
                }
            }
        }
        
        // Fall back to regular DMA allocation
        let buffer = self.osal.dma_alloc(size, flags.base_flags)
            .map_err(|_| RknpuError::OutOfMemory)?;
        
        let mut mem_obj = MemoryObject::new(handle, buffer, flags);
        
        // Create scatter-gather table if needed
        if flags.iommu {
            let mut sgt = ScatterGatherTable::new();
            sgt.add_entry(mem_obj.buffer.dma_addr, size as u32);
            mem_obj.sgt = Some(sgt);
        }
        
        self.objects.push(mem_obj);
        
        self.osal.log_debug(&alloc::format!("DMA allocated: handle={}, vaddr={:p}, paddr=0x{:x}, size=0x{:x}",
                                           handle, mem_obj.buffer.virt_addr.as_ptr(), 
                                           mem_obj.buffer.phys_addr, size));
        
        Ok(handle)
    }
    
    /// Free memory object
    pub fn free(&mut self, handle: MemHandle) -> Result<(), RknpuError> {
        let index = self.objects.iter().position(|obj| obj.handle == handle)
            .ok_or(RknpuError::InvalidParameter)?;
        
        let mut obj = self.objects.remove(index);
        
        // Release reference and check if we should actually free
        if !obj.release() {
            // Still has references, put it back
            self.objects.insert(index, obj);
            return Ok(());
        }
        
        self.osal.log_debug(&alloc::format!("Freeing memory object: handle={}", handle));
        
        // Handle SRAM deallocation
        if obj.flags.sram {
            if let Some(ref mut sram_mgr) = self.sram_manager {
                let chunks = (obj.buffer.size + sram_mgr.chunk_size - 1) / sram_mgr.chunk_size;
                sram_mgr.free_chunks(obj.buffer.phys_addr, chunks);
                return Ok(());
            }
        }
        
        // Handle NBUF deallocation (no-op for now as it's a fixed mapping)
        if obj.flags.nbuf {
            return Ok(());
        }
        
        // Free regular DMA buffer
        self.osal.dma_free(obj.buffer)
            .map_err(|_| RknpuError::DeviceError)?;
        
        Ok(())
    }
    
    /// Get memory object by handle
    pub fn get_object(&self, handle: MemHandle) -> Result<&MemoryObject, RknpuError> {
        self.objects.iter()
            .find(|obj| obj.handle == handle)
            .ok_or(RknpuError::InvalidParameter)
    }
    
    /// Get mutable memory object by handle
    pub fn get_object_mut(&mut self, handle: MemHandle) -> Result<&mut MemoryObject, RknpuError> {
        self.objects.iter_mut()
            .find(|obj| obj.handle == handle)
            .ok_or(RknpuError::InvalidParameter)
    }
    
    /// Synchronize memory object
    pub fn sync(&self, handle: MemHandle, direction: DmaSyncDirection) -> Result<(), RknpuError> {
        let obj = self.get_object(handle)?;
        
        self.osal.dma_sync(&obj.buffer, direction)
            .map_err(|_| RknpuError::DeviceError)?;
        
        self.osal.log_debug(&alloc::format!("Memory synced: handle={}, direction={:?}", handle, direction));
        
        Ok(())
    }
    
    /// Add reference to memory object
    pub fn add_ref(&mut self, handle: MemHandle) -> Result<(), RknpuError> {
        let obj = self.get_object_mut(handle)?;
        obj.add_ref();
        Ok(())
    }
    
    /// Map memory object to userspace (placeholder)
    pub fn map_to_userspace(&self, handle: MemHandle) -> Result<u64, RknpuError> {
        let obj = self.get_object(handle)?;
        // In a real implementation, this would create a VMA mapping
        // For now, just return a fake offset
        Ok(handle as u64 * 0x1000)
    }
    
    /// Get SRAM usage statistics
    pub fn get_sram_stats(&self) -> (usize, usize) {
        if let Some(ref sram_mgr) = self.sram_manager {
            (sram_mgr.get_total_size(), sram_mgr.get_free_size())
        } else {
            (0, 0)
        }
    }
    
    /// Get total number of allocated objects
    pub fn get_object_count(&self) -> usize {
        self.objects.len()
    }
    
    /// Check if address is within NBUF range
    pub fn is_nbuf_address(&self, addr: u64) -> bool {
        if let Some(ref nbuf) = self.nbuf_mapping {
            nbuf.is_valid_addr(addr)
        } else {
            false
        }
    }
    
    /// Get memory object virtual address
    pub fn get_virtual_address(&self, handle: MemHandle) -> Result<NonNull<u8>, RknpuError> {
        let obj = self.get_object(handle)?;
        Ok(obj.buffer.virt_addr)
    }
    
    /// Get memory object physical/DMA address
    pub fn get_dma_address(&self, handle: MemHandle) -> Result<u64, RknpuError> {
        let obj = self.get_object(handle)?;
        Ok(obj.buffer.dma_addr)
    }
    
    /// Get memory object size
    pub fn get_size(&self, handle: MemHandle) -> Result<usize, RknpuError> {
        let obj = self.get_object(handle)?;
        Ok(obj.buffer.size)
    }
    
    /// Cleanup all memory objects (for shutdown)
    pub fn cleanup_all(&mut self) {
        self.osal.log_info("Cleaning up all memory objects");
        
        let handles: Vec<MemHandle> = self.objects.iter().map(|obj| obj.handle).collect();
        
        for handle in handles {
            if let Err(e) = self.free(handle) {
                self.osal.log_error(&alloc::format!("Failed to free handle {}: {:?}", handle, e));
            }
        }
        
        self.objects.clear();
        self.next_handle = 1;
    }
}