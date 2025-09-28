#![no_std]
#![no_main]
#![feature(used_with_arg)]

extern crate alloc;
#[macro_use]
extern crate log;
extern crate bare_test;

#[bare_test::tests]
mod tests {
    use core::time::Duration;

    use aarch64_cpu_ext::asm::{
        barrier::{SY, dsb, isb},
        tlb::{VMALLE1, VMALLE1IS, tlbi},
    };
    use alloc::vec::Vec;
    use bare_test::{
        fdt_parser::Node,
        globals::{PlatformInfoKind, global_val},
        mem::{iomap, page_size},
        time::sleep,
    };
    use rknpu::{Rknpu, RknpuConfig, RknpuType};

    #[test]
    fn it_works() {
        let mut npu = find_rknpu();
        npu.open().unwrap();
        info!("Opened RKNPU");

        info!("Found RKNPU {:#x}", npu.get_hw_version());
    }

    fn find_rknpu() -> Rknpu {
        let PlatformInfoKind::DeviceTree(fdt) = &global_val().platform_info;
        let fdt = fdt.get();

        let node = fdt
            .find_compatible(&["rockchip,rk3588-rknpu"])
            .next()
            .unwrap();

        info!("Found node: {}", node.name());
        let mut config = None;
        for c in node.compatibles() {
            if c == "rockchip,rk3588-rknpu" {
                config = Some(RknpuConfig {
                    rknpu_type: RknpuType::Rk3588,
                });
                break;
            }
        }

        let config = config.expect("Unsupported RKNPU compatible");

        let regs = node.reg().unwrap();

        let mut base_regs = Vec::new();

        for reg in regs {
            base_regs.push(iomap(
                (reg.address as usize).into(),
                reg.size.unwrap_or(page_size()),
            ));
        }

        let t1 = 0xfd7c08ec_usize;
        let v = 0xfd7c0000_usize;
        let of = t1 - v;
        let p = iomap(v.into(), 0x1000);
        unsafe {
            let p = p.add(of);
            debug!("test addr {:#p}", p);
            let v = p.read_volatile();
            debug!("test read {:#x}", v);
        }

        Rknpu::new(&base_regs, config)
    }
}
