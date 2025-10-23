use alloc::collections::btree_map::BTreeMap;
use dma_api::{DVec, Direction};
use spin::Mutex;

use crate::ioctrl::{RknpuMemCreate, RknpuMemSync};

pub struct GemPool {
    pool: BTreeMap<u32, DVec<u8>>,
}

impl GemPool {
    pub const fn new() -> Self {
        GemPool {
            pool: BTreeMap::new(),
        }
    }

    pub fn create(&mut self, args: &mut RknpuMemCreate) {
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
