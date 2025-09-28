//! Operating System Abstraction Layer (OSAL) for RKNPU device layer
//!
//! This module provides platform-agnostic abstractions for system-dependent operations
//! such as memory management, time operations, and synchronization primitives.

use alloc::vec::Vec;
use core::ptr::NonNull;

/// Physical address type
pub type PhysAddr = u64;

/// DMA address type  
pub type DmaAddr = u64;

/// Time type for timestamps
pub type TimeStamp = u64;

/// Error types for OSAL operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsalError {
    OutOfMemory,
    InvalidParameter,
    TimeoutError,
    DeviceError,
    NotSupported,
}

/// Memory allocation flags
#[derive(Debug, Clone, Copy)]
pub struct MemoryFlags {
    pub cacheable: bool,
    pub contiguous: bool,
    pub zeroing: bool,
    pub dma32: bool,
}

impl Default for MemoryFlags {
    fn default() -> Self {
        Self {
            cacheable: false,
            contiguous: true,
            zeroing: true,
            dma32: false,
        }
    }
}

/// Memory buffer descriptor
#[derive(Debug)]
pub struct MemoryBuffer {
    pub virt_addr: NonNull<u8>,
    pub phys_addr: PhysAddr,
    pub dma_addr: DmaAddr,
    pub size: usize,
    pub flags: MemoryFlags,
}

/// DMA synchronization direction
#[derive(Debug, Clone, Copy)]
pub enum DmaSyncDirection {
    ToDevice,
    FromDevice,
    Bidirectional,
}

/// OSAL trait for platform-specific implementations
pub trait Osal {
    /// Get current timestamp in microseconds
    fn get_time_us(&self) -> TimeStamp;

    /// Sleep for specified microseconds
    fn udelay(&self, us: u32);

    /// Sleep for specified milliseconds
    fn msleep(&self, ms: u32);

    /// Check if timeout occurred
    fn timeout_check(&self, start_time: TimeStamp, timeout_us: u32) -> bool {
        let elapsed = self.get_time_us().saturating_sub(start_time);
        elapsed >= timeout_us as u64
    }
}

/// Scatter-gather list entry
#[derive(Debug, Clone)]
pub struct ScatterGatherEntry {
    pub addr: DmaAddr,
    pub length: u32,
}

/// Scatter-gather table
#[derive(Debug, Clone)]
pub struct ScatterGatherTable {
    pub entries: Vec<ScatterGatherEntry>,
}

impl ScatterGatherTable {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, addr: DmaAddr, length: u32) {
        self.entries.push(ScatterGatherEntry { addr, length });
    }

    pub fn total_size(&self) -> u64 {
        self.entries.iter().map(|e| e.length as u64).sum()
    }
}

/// Memory mapping for NBUF (Neural Buffer)
#[derive(Debug)]
pub struct NBufMapping {
    pub phys_start: PhysAddr,
    pub size: usize,
    pub virt_base: Option<NonNull<u8>>,
}

impl NBufMapping {
    pub fn new(phys_start: PhysAddr, size: usize) -> Self {
        Self {
            phys_start,
            size,
            virt_base: None,
        }
    }

    pub fn is_valid_addr(&self, addr: PhysAddr) -> bool {
        addr >= self.phys_start && addr < self.phys_start + self.size as u64
    }
}

/// SRAM memory manager
#[derive(Debug)]
pub struct SramManager {
    pub start_addr: PhysAddr,
    pub total_size: usize,
    pub chunk_size: usize,
    pub free_chunks: usize,
    pub total_chunks: usize,
}

impl SramManager {
    pub fn new(start_addr: PhysAddr, total_size: usize, chunk_size: usize) -> Self {
        let total_chunks = total_size / chunk_size;
        Self {
            start_addr,
            total_size,
            chunk_size,
            free_chunks: total_chunks,
            total_chunks,
        }
    }

    pub fn allocate_chunks(&mut self, num_chunks: usize) -> Option<PhysAddr> {
        if self.free_chunks >= num_chunks {
            self.free_chunks -= num_chunks;
            // 简化的分配逻辑，实际实现需要更复杂的管理
            Some(
                self.start_addr
                    + (self.total_chunks - self.free_chunks - num_chunks) as u64
                        * self.chunk_size as u64,
            )
        } else {
            None
        }
    }

    pub fn free_chunks(&mut self, _addr: PhysAddr, num_chunks: usize) {
        self.free_chunks += num_chunks;
        if self.free_chunks > self.total_chunks {
            self.free_chunks = self.total_chunks;
        }
    }

    pub fn get_free_size(&self) -> usize {
        self.free_chunks * self.chunk_size
    }

    pub fn get_total_size(&self) -> usize {
        self.total_chunks * self.chunk_size
    }
}
