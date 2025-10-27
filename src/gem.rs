use alloc::collections::btree_map::BTreeMap;
use dma_api::{DVec, Direction};

use crate::{
    RknpuError,
    ioctrl::{RknpuMemCreate, RknpuMemSync},
};

pub struct GemPool {
    pool: BTreeMap<u32, DVec<u8>>,
}

impl GemPool {
    pub const fn new() -> Self {
        GemPool {
            pool: BTreeMap::new(),
        }
    }

    pub fn create(&mut self, args: &mut RknpuMemCreate) -> Result<(), RknpuError> {
        let data = DVec::zeros(
            u32::MAX as _,
            args.size as _,
            0x1000,
            Direction::Bidirectional,
        )
        .unwrap();
        args.handle = data.bus_addr() as _;
        args.sram_size = data.len() as _;
        args.dma_addr = data.bus_addr();
        self.pool.insert(args.handle, data);
        Ok(())
    }

    /// Get the physical address and size of the memory object.
    pub fn get_phys_addr_and_size(&self, handle: u32) -> Option<(u64, usize)> {
        self.pool
            .get(&handle)
            .map(|dvec| (dvec.bus_addr(), dvec.len()))
    }

    pub fn sync(&mut self, args: &mut RknpuMemSync) {}

    pub fn destroy(&mut self, handle: u32) {
        self.pool.remove(&handle);
    }
}

impl Default for GemPool {
    fn default() -> Self {
        Self::new()
    }
}
