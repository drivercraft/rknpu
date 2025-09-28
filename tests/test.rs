#![no_std]
#![no_main]
#![feature(used_with_arg)]

extern crate alloc;
#[macro_use]
extern crate log;
extern crate bare_test;

#[bare_test::tests]
mod tests {
    use bare_test::{
        fdt_parser::Node,
        globals::{PlatformInfoKind, global_val},
        mem::{iomap, page_size},
    };
    use rknpu::{Rknpu, RknpuConfig, RknpuType};

    #[test]
    fn it_works() {
        let npu = find_rknpu();
    }

    fn find_rknpu() -> Rknpu {
        let PlatformInfoKind::DeviceTree(fdt) = &global_val().platform_info;
        let fdt = fdt.get();

        let node = fdt.find_compatible(&["rockchip,rk3588-rknpu"]).next().unwrap();

        info!("Found node: {}", node.name());
        let mut config = None;
        for c in node.compatibles() {
            if c == "rockchip,rk3588-rknpu" {
                config = Some(RknpuConfig::new(RknpuType::Rk3588));
                break;
            }
        }

        let config = config.expect("Unsupported RKNPU compatible");

        let reg = node
            .reg()
            .unwrap()
            .next()
            .expect("RKNPU node has no reg property");

        let addr = iomap(
            (reg.address as usize).into(),
            reg.size.unwrap_or(page_size()),
        );

        Rknpu::new(addr, config)
    }
}
