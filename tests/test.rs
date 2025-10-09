#![no_std]
#![no_main]
#![feature(used_with_arg)]

extern crate alloc;
#[macro_use]
extern crate log;
extern crate bare_test;

#[bare_test::tests]
mod tests {
    use core::ptr::NonNull;

    use alloc::vec::Vec;
    use bare_test::{
        globals::{PlatformInfoKind, global_val},
        mem::{iomap, page_size},
    };
    use rknpu::{Rknpu, RknpuConfig, RknpuType};
    use rockchip_pm::{PD, RkBoard, RockchipPM};

    /// NPU 主电源域
    pub const NPU: PD = PD(8);
    /// NPU TOP 电源域  
    pub const NPUTOP: PD = PD(9);
    /// NPU1 电源域
    pub const NPU1: PD = PD(10);
    /// NPU2 电源域
    pub const NPU2: PD = PD(11);

    #[test]
    fn it_works() {
        let reg = get_syscon_addr();
        let board = RkBoard::Rk3588;

        let mut pm = RockchipPM::new(reg, board);

        pm.power_domain_on(NPUTOP).unwrap();
        pm.power_domain_on(NPU).unwrap();
        pm.power_domain_on(NPU1).unwrap();
        pm.power_domain_on(NPU2).unwrap();

        info!("Powered on NPU domains");

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
            let start_raw = reg.address as usize;
            let end = start_raw + reg.size.unwrap_or(page_size());

            let start = start_raw & !(page_size() - 1);
            let offset = start_raw - start;
            let end = (end + page_size() - 1) & !(page_size() - 1);
            let size = end - start;

            base_regs.push(unsafe { iomap(start.into(), size).add(offset) });
        }

        Rknpu::new(&base_regs, config)
    }

    fn get_syscon_addr() -> NonNull<u8> {
        let PlatformInfoKind::DeviceTree(fdt) = &global_val().platform_info;
        let fdt = fdt.get();

        let node = fdt
            .find_compatible(&["syscon"])
            .find(|n| n.name().contains("power-manage"))
            .expect("Failed to find syscon node");

        info!("Found node: {}", node.name());

        let regs = node.reg().unwrap().collect::<Vec<_>>();
        let start = regs[0].address as usize;
        let end = start + regs[0].size.unwrap_or(0);
        info!("Syscon address range: 0x{:x} - 0x{:x}", start, end);
        let start = start & !(page_size() - 1);
        let end = (end + page_size() - 1) & !(page_size() - 1);
        info!("Aligned Syscon address range: 0x{:x} - 0x{:x}", start, end);
        iomap(start.into(), end - start)
    }
}
