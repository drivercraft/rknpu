//! Device interface layer for RKNPU
//! 
//! This module provides the main device interface that combines hardware
//! abstraction layer, memory management, and OSAL abstractions.

use core::ptr::NonNull;
use alloc::vec::Vec;
use crate::osal::{Osal, DmaSyncDirection};
use crate::hal::{HardwareLayer, TaskSubmission, TaskFlags, SubcoreTask, HardwareJob, JobState};
use crate::memory::{MemoryManager, NpuMemoryFlags, MemHandle};
use crate::config::RknpuConfig;
use crate::err::RknpuError;

/// Device action types (matching kernel driver)
#[derive(Debug, Clone, Copy)]
pub enum DeviceAction {
    GetHwVersion,
    GetDrvVersion,
    GetFreq,
    SetFreq,
    GetVolt,
    SetVolt,
    Reset,
    GetBwPriority,
    SetBwPriority,
    ClearTotalRwAmount,
    GetDtWrAmount,
    GetDtRdAmount,
    GetWtRdAmount,
    GetTotalRwAmount,
    GetIommuEn,
    SetProcNice,
    GetTotalSramSize,
    GetFreeSramSize,
}

/// Device information structure
#[derive(Debug)]
pub struct DeviceInfo {
    pub hw_version: u32,
    pub drv_version: u32,
    pub core_count: usize,
    pub iommu_enabled: bool,
    pub sram_total_size: usize,
    pub sram_free_size: usize,
}

/// RKNPU device interface
pub struct RknpuDevice<O: Osal> {
    hal: HardwareLayer<O>,
    memory: MemoryManager<O>,
    config: RknpuConfig,
    device_info: DeviceInfo,
    initialized: bool,
}

impl<O: Osal> RknpuDevice<O> {
    /// Create new RKNPU device instance
    pub fn new(base_addrs: Vec<NonNull<u8>>, config: RknpuConfig, osal: O) -> Result<Self, RknpuError> {
        if base_addrs.is_empty() {
            return Err(RknpuError::InvalidParameter);
        }
        
        let iommu_enabled = true; // Assume IOMMU is enabled for now
        
        let hal = HardwareLayer::new(base_addrs, config.clone(), osal.clone())?;
        let memory = MemoryManager::new(osal, iommu_enabled);
        
        let device_info = DeviceInfo {
            hw_version: 0,
            drv_version: 0x010000, // v1.0.0
            core_count: config.num_irqs,
            iommu_enabled,
            sram_total_size: 0,
            sram_free_size: 0,
        };
        
        Ok(Self {
            hal,
            memory,
            config,
            device_info,
            initialized: false,
        })
    }
    
    /// Initialize the device
    pub fn initialize(&mut self) -> Result<(), RknpuError> {
        if self.initialized {
            return Ok(());
        }
        
        self.hal.osal.log_info("Initializing RKNPU device");
        
        // Initialize hardware layer
        self.hal.initialize()?;
        
        // Get hardware version
        self.device_info.hw_version = self.hal.get_hw_version()?;
        
        // Initialize SRAM if available
        if self.config.nbuf_size > 0 {
            self.memory.init_nbuf(self.config.nbuf_phyaddr, self.config.nbuf_size as usize)?;
        }
        
        self.initialized = true;
        self.hal.osal.log_info("RKNPU device initialized successfully");
        
        Ok(())
    }
    
    /// Shutdown the device
    pub fn shutdown(&mut self) {
        if !self.initialized {
            return;
        }
        
        self.hal.osal.log_info("Shutting down RKNPU device");
        
        // Cleanup all memory objects
        self.memory.cleanup_all();
        
        // Reset hardware state
        let _ = self.hal.soft_reset();
        
        self.initialized = false;
        self.hal.osal.log_info("RKNPU device shutdown completed");
    }
    
    /// Execute device action
    pub fn execute_action(&mut self, action: DeviceAction, value: &mut u32) -> Result<(), RknpuError> {
        if !self.initialized {
            return Err(RknpuError::DeviceNotReady);
        }
        
        match action {
            DeviceAction::GetHwVersion => {
                *value = self.device_info.hw_version;
                Ok(())
            },
            DeviceAction::GetDrvVersion => {
                *value = self.device_info.drv_version;
                Ok(())
            },
            DeviceAction::GetIommuEn => {
                *value = if self.device_info.iommu_enabled { 1 } else { 0 };
                Ok(())
            },
            DeviceAction::Reset => {
                self.hal.soft_reset()
            },
            DeviceAction::ClearTotalRwAmount => {
                self.hal.clear_rw_amount()
            },
            DeviceAction::GetDtWrAmount => {
                let (dt_wr, _, _) = self.hal.get_rw_amount()?;
                *value = dt_wr;
                Ok(())
            },
            DeviceAction::GetDtRdAmount => {
                let (_, dt_rd, _) = self.hal.get_rw_amount()?;
                *value = dt_rd;
                Ok(())
            },
            DeviceAction::GetWtRdAmount => {
                let (_, _, wt_rd) = self.hal.get_rw_amount()?;
                *value = wt_rd;
                Ok(())
            },
            DeviceAction::GetTotalRwAmount => {
                let (dt_wr, dt_rd, wt_rd) = self.hal.get_rw_amount()?;
                *value = dt_wr + dt_rd + wt_rd;
                Ok(())
            },
            DeviceAction::GetTotalSramSize => {
                let (total, _) = self.memory.get_sram_stats();
                *value = total as u32;
                Ok(())
            },
            DeviceAction::GetFreeSramSize => {
                let (_, free) = self.memory.get_sram_stats();
                *value = free as u32;
                Ok(())
            },
            _ => {
                self.hal.osal.log_warn(&alloc::format!("Unsupported action: {:?}", action));
                Err(RknpuError::NotSupported)
            }
        }
    }
    
    /// Allocate memory
    pub fn memory_create(&mut self, size: usize, flags: NpuMemoryFlags) -> Result<MemHandle, RknpuError> {
        if !self.initialized {
            return Err(RknpuError::DeviceNotReady);
        }
        
        self.memory.allocate(size, flags)
    }
    
    /// Free memory
    pub fn memory_destroy(&mut self, handle: MemHandle) -> Result<(), RknpuError> {
        self.memory.free(handle)
    }
    
    /// Synchronize memory
    pub fn memory_sync(&self, handle: MemHandle, direction: DmaSyncDirection) -> Result<(), RknpuError> {
        self.memory.sync(handle, direction)
    }
    
    /// Map memory to userspace (returns fake offset)
    pub fn memory_map(&self, handle: MemHandle) -> Result<u64, RknpuError> {
        self.memory.map_to_userspace(handle)
    }
    
    /// Submit task for execution
    pub fn submit_task(&mut self, submission: TaskSubmission) -> Result<u32, RknpuError> {
        if !self.initialized {
            return Err(RknpuError::DeviceNotReady);
        }
        
        self.hal.submit_task(submission)
    }
    
    /// Handle interrupt for specific core
    pub fn irq_handle(&mut self, core_index: usize) -> Result<(), RknpuError> {
        self.hal.irq_handle(core_index)
    }
    
    /// Get device information
    pub fn get_device_info(&self) -> &DeviceInfo {
        &self.device_info
    }
    
    /// Check if device is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    /// Get memory object virtual address
    pub fn get_memory_vaddr(&self, handle: MemHandle) -> Result<NonNull<u8>, RknpuError> {
        self.memory.get_virtual_address(handle)
    }
    
    /// Get memory object DMA address
    pub fn get_memory_dma_addr(&self, handle: MemHandle) -> Result<u64, RknpuError> {
        self.memory.get_dma_address(handle)
    }
    
    /// Get memory object size
    pub fn get_memory_size(&self, handle: MemHandle) -> Result<usize, RknpuError> {
        self.memory.get_size(handle)
    }
    
    /// Get hardware configuration
    pub fn get_config(&self) -> &RknpuConfig {
        &self.config
    }
    
    /// Get number of active memory objects
    pub fn get_memory_object_count(&self) -> usize {
        self.memory.get_object_count()
    }
    
    /// Create task submission from parameters
    pub fn create_task_submission(
        &self,
        task_handle: MemHandle,
        task_start: u32,
        task_number: u32,
        timeout_ms: u32,
        core_mask: u32,
        flags: TaskFlags,
    ) -> Result<TaskSubmission, RknpuError> {
        let task_buffer = {
            let obj = self.memory.get_object(task_handle)?;
            obj.buffer.clone()
        };
        
        let task_base_addr = self.get_memory_dma_addr(task_handle)?;
        
        // Create subcore tasks for multi-core systems
        let mut subcore_tasks = Vec::new();
        if self.config.num_irqs > 1 {
            // Distribute tasks across cores (simplified)
            let tasks_per_core = task_number / self.config.num_irqs as u32;
            let remaining_tasks = task_number % self.config.num_irqs as u32;
            
            for i in 0..self.config.num_irqs {
                let start = task_start + i as u32 * tasks_per_core;
                let num = tasks_per_core + if i < remaining_tasks as usize { 1 } else { 0 };
                
                subcore_tasks.push(SubcoreTask {
                    task_start: start,
                    task_number: num,
                });
            }
        }
        
        Ok(TaskSubmission {
            flags,
            timeout_ms,
            task_start,
            task_number,
            priority: 0,
            task_buffer,
            task_base_addr,
            core_mask,
            subcore_tasks,
        })
    }
}

// Implement Drop to ensure cleanup
impl<O: Osal> Drop for RknpuDevice<O> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// Make the device Send + Sync if the OSAL is
unsafe impl<O: Osal + Send> Send for RknpuDevice<O> {}
unsafe impl<O: Osal + Sync> Sync for RknpuDevice<O> {}