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
    use arm_scmi::{Scmi, Shmem, Smc};
    use bare_test::{
        globals::{global_val, PlatformInfoKind}, irq::Phandle, mem::{iomap, page_size}
    };
    use num_align::NumAlign;
    use rk3588_clk::{
        Rk3588Cru,
        constant::{
            ACLK_NPU0, ACLK_NPU1, ACLK_NPU2, CLK_CORE_NPU_PVTM, CLK_NPU_CM0_RTC, CLK_NPU_DSU0,
            CLK_NPU_PVTM, CLK_NPUTIMER_ROOT, CLK_NPUTIMER0, CLK_NPUTIMER1, FCLK_NPU_CM0_CORE,
            HCLK_NPU_CM0_ROOT, HCLK_NPU_ROOT, HCLK_NPU0, HCLK_NPU1, HCLK_NPU2, PCLK_NPU_GRF,
            PCLK_NPU_PVTM, PCLK_NPU_ROOT, PCLK_NPU_TIMER, PCLK_NPU_WDT, TCLK_NPU_WDT,
        },
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
        set_up_scmi();

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

        matmul::run(&mut npu).expect("matmul workload failed");
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
        let clk_ls = node.clocks().collect::<Vec<_>>();
        let mut clk_ctrl = configure_npu_clocks();
        info!("Configured NPU clock tree");
        for clk in &clk_ls {
            info!("Clock: {:?}", clk);
            if clk.node.name().contains("protocol") {
                continue;
            }
            clk_ctrl.npu_gate_enable(clk.select as _).unwrap();
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

    fn configure_npu_clocks() -> Rk3588Cru {
        let cru_addr = get_cru_addr();
        Rk3588Cru::new(cru_addr)
        // let cru = Rk3588Cru::new(cru_addr);

        // // Program the primary NPU clock tree to known-good defaults. Ignore failures for now.
        // let _ = cru.npu_set_clk(HCLK_NPU_ROOT, 200_000_000);
        // let _ = cru.npu_set_clk(CLK_NPU_DSU0, 800_000_000);
        // let _ = cru.npu_set_clk(PCLK_NPU_ROOT, 100_000_000);
        // let _ = cru.npu_set_clk(HCLK_NPU_CM0_ROOT, 200_000_000);
        // let _ = cru.npu_set_clk(CLK_NPU_CM0_RTC, 24_000_000);
        // let _ = cru.npu_set_clk(CLK_NPUTIMER_ROOT, 100_000_000);

        // // Ensure the essential gates are open.
        // for gate in [
        //     ACLK_NPU0,
        //     HCLK_NPU0,
        //     ACLK_NPU1,
        //     HCLK_NPU1,
        //     ACLK_NPU2,
        //     HCLK_NPU2,
        //     PCLK_NPU_PVTM,
        //     PCLK_NPU_GRF,
        //     CLK_NPU_PVTM,
        //     CLK_CORE_NPU_PVTM,
        //     PCLK_NPU_TIMER,
        //     CLK_NPUTIMER0,
        //     CLK_NPUTIMER1,
        //     PCLK_NPU_WDT,
        //     TCLK_NPU_WDT,
        //     FCLK_NPU_CM0_CORE,
        // ] {
        //     if let Err(err) = cru.npu_gate_enable(gate) {
        //         warn!("Failed to enable gate {gate}: {err}");
        //     }
        // }
    }

    fn get_cru_addr() -> NonNull<u8> {
        let PlatformInfoKind::DeviceTree(fdt) = &global_val().platform_info;
        let fdt = fdt.get();

        let node = fdt
            .find_compatible(&["rockchip,rk3588-cru"])
            .next()
            .expect("Failed to find CRU node");

        info!("Found node: {}", node.name());

        let reg = node
            .reg()
            .and_then(|mut regs| regs.next())
            .expect("CRU node missing reg range");

        let start_raw = reg.address as usize;
        let size = reg.size.unwrap_or(page_size());

        let start = start_raw & !(page_size() - 1);
        let end = (start_raw + size + page_size() - 1) & !(page_size() - 1);
        let offset = start_raw - start;

        let mapping = iomap(start.into(), end - start);
        let ptr = unsafe { mapping.as_ptr().add(offset) };

        // SAFETY: iomap guarantees a valid mapping; offset is within bounds.
        unsafe { NonNull::new_unchecked(ptr) }
    }

    fn set_up_scmi() {
        let PlatformInfoKind::DeviceTree(fdt) = &global_val().platform_info;
        let fdt = fdt.get();
        let node = fdt
            .find_compatible(&["arm,scmi-smc"])
            .next()
            .expect("scmi not found");

        info!("found scmi node: {:?}", node.name());

        let shmem_ph: Phandle = node
            .find_property("shmem")
            .expect("shmem property not found")
            .u32()
            .into();

        let shmem_node = fdt
            .get_node_by_phandle(shmem_ph)
            .expect("shmem node not found");

        info!("found shmem node: {:?}", shmem_node.name());

        let shmem_reg = shmem_node.reg().unwrap().collect::<Vec<_>>();
        assert_eq!(shmem_reg.len(), 1);
        let shmem_reg = shmem_reg[0];
        let shmem_addr = iomap(
            (shmem_reg.address as usize).into(),
            shmem_reg.size.unwrap().align_up(0x1000),
        );

        let func_id = node
            .find_property("arm,smc-id")
            .expect("function-id property not found")
            .u32();

        info!("shmem reg: {:?}", shmem_reg);
        info!("func_id: {:#x}", func_id);

        let irq_num = node.find_property("a2p").map(|irq_prop| irq_prop.u32());

        let shmem = Shmem {
            address: shmem_addr,
            bus_address: shmem_reg.child_bus_address as usize,
            size: shmem_reg.size.unwrap(),
        };
        let kind = Smc::new(func_id, irq_num);
        let scmi = Scmi::new(kind, shmem);

        let mut pclk = scmi.protocol_clk();

        let ls = [
            (0u32, "clk0", 0x30a32c00),
            (2u32, "clk1", 0x30a32c00),
            (3u32, "clk2", 0x30a32c00),
        ];
        for (id, name, clk) in ls {
            pclk.clk_enable(id).unwrap();
            let rate = pclk.rate_get(id).unwrap();
            info!("Clock {} (id={}): rate={} Hz", name, id, rate);
            pclk.rate_set(id, clk).unwrap();
            let rate = pclk.rate_get(id).unwrap();
            info!("Clock {} (id={}): new rate={} Hz", name, id, rate);
        }
    }

    mod matmul {
        use alloc::{vec, vec::Vec};
        use core::{array, convert::TryFrom, fmt, mem, ptr::NonNull, sync::atomic::AtomicU32};
        use spin::Once;

        use dma_api::{DError, DVec, Direction, Osal};
        use rknpu::{
            RKNPU_CORE0_MASK, RKNPU_JOB_PC, RKNPU_JOB_PINGPONG, RKNPU_MAX_CORES,
            RKNPU_MAX_SUBCORE_TASKS, Rknpu, RknpuAction, RknpuError, RknpuJob, RknpuSubcoreTask,
            RknpuSubmit, RknpuTask,
        };

        const MATMUL_DIMS: MatmulDims = MatmulDims { m: 4, k: 32, n: 16 };
        const REGCMD_WORDS: usize = 112;
        const TASK_SIZE: usize = mem::size_of::<RknpuTask>();
        const DMA_MASK: u64 = u32::MAX as _;
        const DMA_ALIGN: usize = 0x1000;
        const BUFFER_ALIGN: usize = 0x40;
        const PC_DATA_EXTRA_AMOUNT: u32 = 4;
        const INPUT_PACKING: usize = 16;
        const OUTPUT_PACKING: usize = 4;

        pub fn run(npu: &mut Rknpu) -> Result<(), MatmulError> {
            let _ = npu.action(RknpuAction::ActReset);

            let mut regcmd = dma_vec::<u64>(REGCMD_WORDS, DMA_ALIGN, Direction::ToDevice).unwrap();
            let mut task_obj = dma_vec::<u8>(TASK_SIZE, DMA_ALIGN, Direction::ToDevice).unwrap();
            let mut input = dma_vec::<i8>(
                MATMUL_DIMS.m * MATMUL_DIMS.k,
                BUFFER_ALIGN,
                Direction::ToDevice,
            )
            .unwrap();

            let mut weights = dma_vec::<i8>(
                MATMUL_DIMS.n * MATMUL_DIMS.k,
                BUFFER_ALIGN,
                Direction::ToDevice,
            )
            .unwrap();
            let output = dma_vec::<i32>(
                MATMUL_DIMS.m * MATMUL_DIMS.n,
                BUFFER_ALIGN,
                Direction::FromDevice,
            )
            .unwrap();

            log_dma_buffers(&regcmd, &task_obj, &input, &weights, &output);

            let reference = prepare_reference(MATMUL_DIMS);
            populate_weights(&reference, &mut weights);
            populate_input(&reference, &mut input);

            let params = MatmulParams::new(
                MATMUL_DIMS,
                input.bus_addr(),
                weights.bus_addr(),
                output.bus_addr(),
            )?;
            let ops = gen_matmul_int8(&params).unwrap();
            regcmd.copy_from_slice(&ops);

            let regcfg_amount = (REGCMD_WORDS as u32).saturating_sub(PC_DATA_EXTRA_AMOUNT + 4);
            const PC_INT_MASK: u32 = 0xD4;

            let task = RknpuTask {
                flags: 0,
                op_idx: 0,
                enable_mask: 0xD,
                int_mask: PC_INT_MASK,
                int_clear: 0x1_FFFF,
                int_status: 0,
                regcfg_amount,
                regcfg_offset: 0,
                regcmd_addr: regcmd.bus_addr(),
            };
            let task_int_mask = task.int_mask;
            let task_bytes = unsafe {
                core::slice::from_raw_parts((&task as *const RknpuTask).cast::<u8>(), TASK_SIZE)
            };
            task_obj.copy_from_slice(task_bytes);

            let mut submit = RknpuSubmit {
                flags: RKNPU_JOB_PC | RKNPU_JOB_PINGPONG,
                timeout: 6000,
                task_start: 0,
                task_number: 1,
                task_counter: 0,
                priority: 0,
                task_obj,
                iommu_domain_id: 0,
                reserved: 0,
                task_base_addr: 0,
                hw_elapse_time: 0,
                core_mask: RKNPU_CORE0_MASK,
                fence_fd: -1,
                subcore_task: [RknpuSubcoreTask::default(); RKNPU_MAX_SUBCORE_TASKS],
            };
            submit.subcore_task[0] = RknpuSubcoreTask {
                task_start: 0,
                task_number: 1,
            };

            let mut job = RknpuJob {
                use_core_num: 1,
                args: submit,
                first_task: 0,
                last_task: 0,
                int_mask: [0; RKNPU_MAX_CORES],
                int_status: [0; RKNPU_MAX_CORES],
                submit_count: array::from_fn(|_| AtomicU32::new(0)),
            };

            npu.commit_job(&mut job)
                .map_err(MatmulError::Submit)
                .unwrap();
            npu.wait_for_completion(0, task_int_mask, 1_000_000)
                .map_err(MatmulError::Wait)
                .unwrap();

            output.preper_read_all();

            verify_result(&reference, &output);

            info!(
                "Matmul {}x{}x{} completed successfully",
                MATMUL_DIMS.m, MATMUL_DIMS.k, MATMUL_DIMS.n
            );
            Ok(())
        }

        fn populate_weights(reference: &MatmulReference, weights: &mut DVec<i8>) {
            for n in 1..=MATMUL_DIMS.n {
                for k in 1..=MATMUL_DIMS.k {
                    let idx = weight_int8(MATMUL_DIMS.k, n, k);
                    weights.set(idx, reference.b[(n - 1) * MATMUL_DIMS.k + (k - 1)]);
                }
            }
        }

        fn populate_input(reference: &MatmulReference, input: &mut DVec<i8>) {
            for m in 1..=MATMUL_DIMS.m {
                for k in 1..=MATMUL_DIMS.k {
                    let idx = feature_data(MATMUL_DIMS.k, MATMUL_DIMS.m, 1, INPUT_PACKING, k, m, 1);
                    input.set(idx, reference.a[(m - 1) * MATMUL_DIMS.k + (k - 1)]);
                }
            }
        }

        fn verify_result(reference: &MatmulReference, output: &DVec<i32>) {
            for m in 1..=MATMUL_DIMS.m {
                for n in 1..=MATMUL_DIMS.n {
                    let offset =
                        feature_data(MATMUL_DIMS.n, MATMUL_DIMS.m, 1, OUTPUT_PACKING, n, m, 1);
                    let actual = output.get(offset).expect("output index");
                    let expected = reference.expected[(m - 1) * MATMUL_DIMS.n + (n - 1)];
                    assert_eq!(actual, expected, "m={} n={}", m - 1, n - 1);
                }
            }
        }

        fn dma_vec<T>(
            len: usize,
            align: usize,
            direction: Direction,
        ) -> Result<DVec<T>, MatmulError> {
            DVec::zeros(DMA_MASK, len, align, direction).map_err(MatmulError::from)
        }

        fn log_dma_buffers(
            regcmd: &DVec<u64>,
            task_obj: &DVec<u8>,
            input: &DVec<i8>,
            weights: &DVec<i8>,
            output: &DVec<i32>,
        ) {
            info!(
                "DMA buffers: regcmd ptr=0x{:x} bus=0x{:x}, task ptr=0x{:x} bus=0x{:x}, input ptr=0x{:x} bus=0x{:x}, weights ptr=0x{:x} bus=0x{:x}, output ptr=0x{:x} bus=0x{:x}",
                regcmd.as_ptr() as usize,
                regcmd.bus_addr(),
                task_obj.as_ptr() as usize,
                task_obj.bus_addr(),
                input.as_ptr() as usize,
                input.bus_addr(),
                weights.as_ptr() as usize,
                weights.bus_addr(),
                output.as_ptr() as usize,
                output.bus_addr()
            );
        }

        #[derive(Clone, Copy)]
        struct MatmulDims {
            m: usize,
            k: usize,
            n: usize,
        }

        struct MatmulReference {
            a: Vec<i8>,
            b: Vec<i8>,
            expected: Vec<i32>,
        }

        #[derive(Debug)]
        pub enum MatmulError {
            Dma(DError),
            AddressOverflow,
            CbufOverflow,
            KernelTooLarge,
            Submit(RknpuError),
            Wait(RknpuError),
        }

        impl fmt::Display for MatmulError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    MatmulError::Dma(err) => write!(f, "DMA error: {err:?}"),
                    MatmulError::AddressOverflow => {
                        write!(f, "DMA address exceeds 32-bit range")
                    }
                    MatmulError::CbufOverflow => write!(f, "CBUF capacity exceeded"),
                    MatmulError::KernelTooLarge => {
                        write!(f, "Kernel footprint exceeds CBUF bank size")
                    }
                    MatmulError::Submit(err) => write!(f, "Job submission failed: {err:?}"),
                    MatmulError::Wait(err) => write!(f, "Job completion wait failed: {err:?}"),
                }
            }
        }

        impl From<DError> for MatmulError {
            fn from(value: DError) -> Self {
                Self::Dma(value)
            }
        }

        fn prepare_reference(dims: MatmulDims) -> MatmulReference {
            let mut a = vec![0i8; dims.m * dims.k];
            let mut b = vec![0i8; dims.n * dims.k];

            for (idx, val) in a.iter_mut().enumerate() {
                *val = (idx as i8).wrapping_mul(3).wrapping_sub(7);
            }
            for (idx, val) in b.iter_mut().enumerate() {
                *val = (idx as i8).wrapping_mul(5).wrapping_add(11);
            }

            let expected = matmul_reference(dims, &a, &b);

            MatmulReference { a, b, expected }
        }

        fn matmul_reference(dims: MatmulDims, a: &[i8], b: &[i8]) -> Vec<i32> {
            let mut out = vec![0i32; dims.m * dims.n];

            for m in 0..dims.m {
                for n in 0..dims.n {
                    let mut acc = 0i32;
                    for k in 0..dims.k {
                        let lhs = a[m * dims.k + k] as i32;
                        let rhs = b[n * dims.k + k] as i32;
                        acc += lhs * rhs;
                    }
                    out[m * dims.n + n] = acc;
                }
            }

            out
        }

        struct MatmulParams {
            m: u16,
            k: u16,
            n: u16,
            input_dma: u32,
            weights_dma: u32,
            output_dma: u32,
        }

        impl MatmulParams {
            fn new(
                dims: MatmulDims,
                input_dma: u64,
                weights_dma: u64,
                output_dma: u64,
            ) -> Result<Self, MatmulError> {
                Ok(Self {
                    m: dims.m as u16,
                    k: dims.k as u16,
                    n: dims.n as u16,
                    input_dma: to_dma32(input_dma)?,
                    weights_dma: to_dma32(weights_dma)?,
                    output_dma: to_dma32(output_dma)?,
                })
            }
        }

        fn to_dma32(addr: u64) -> Result<u32, MatmulError> {
            u32::try_from(addr).map_err(|_| MatmulError::AddressOverflow)
        }

        #[derive(Default)]
        struct CnaDesc {
            conv_mode: u8,
            in_precision: u8,
            proc_precision: u8,
            kernel_groups: u8,
            feature_grains: u16,
            conv_y_stride: u8,
            conv_x_stride: u8,
            datain_width: u16,
            datain_height: u16,
            datain_channel: u16,
            dataout_width: u16,
            dataout_height: u16,
            dataout_atomics: u32,
            weight_bytes: u32,
            weight_bytes_per_kernel: u32,
            weight_width: u8,
            weight_height: u8,
            weight_kernels: u16,
            weight_bank: u8,
            data_bank: u8,
            data_entries: u16,
            data_sign: u8,
            cvt_type: u8,
            cvt_bypass: u8,
            cvt_scale0: u16,
            cvt_scale1: u16,
            cvt_scale2: u16,
            cvt_scale3: u16,
            fc_skip_en: u8,
            data_offset: u16,
            pad_left: u8,
            pad_top: u8,
            feature_base_addr: u32,
            weight_offset: u16,
            weight_burst_len: u8,
            data_burst_len: u8,
            line_stride: u32,
            surf_stride: i32,
            dma_width: u16,
            dma_height: u16,
            dma_channel: u16,
            decompress_addr0: u32,
        }

        #[derive(Default)]
        struct CoreDesc {
            proc_precision: u8,
            qd_en: u8,
            dataout_height: u16,
            dataout_width: u16,
            dataout_channel: u16,
        }

        #[derive(Default)]
        struct DpuDesc {
            burst_len: u8,
            conv_mode: u8,
            output_mode: u8,
            flying_mode: u8,
            out_precision: u8,
            in_precision: u8,
            proc_precision: u8,
            dst_base_addr: u32,
            dst_surf_stride: u32,
            width: u16,
            height: u16,
            channel: u16,
            bs_bypass: u8,
            bs_alu_bypass: u8,
            bs_mul_bypass: u8,
            bs_relu_bypass: u8,
            od_bypass: u8,
            size_e_2: u8,
            size_e_1: u8,
            size_e_0: u8,
            channel_wdma: u16,
            height_wdma: u16,
            width_wdma: u16,
            bn_relu_bypass: u8,
            bn_mul_bypass: u8,
            bn_alu_bypass: u8,
            bn_bypass: u8,
            ew_bypass: u8,
            ew_op_bypass: u8,
            ew_lut_bypass: u8,
            ew_op_cvt_bypass: u8,
            ew_relu_bypass: u8,
            fp32tofp16_en: u8,
            out_cvt_scale: u16,
            surf_add: u32,
        }

        #[inline]
        fn div_ceil_u32(value: u32, base: u32) -> u32 {
            debug_assert!(base > 0);
            if value == 0 {
                0
            } else {
                ((value - 1) / base) + 1
            }
        }

        fn gen_matmul_int8(params: &MatmulParams) -> Result<[u64; REGCMD_WORDS], MatmulError> {
            const NPU_CBUF_BANK_SIZE: u32 = 32_768;
            const NPU_CBUF_BANKS: u32 = 12;
            const DIRECT_CONVOLUTION: u8 = 0;
            const PRECISION_INT8: u8 = 0;
            const PRECISION_INT32: u8 = 4;

            let mut cna = CnaDesc::default();
            cna.conv_mode = DIRECT_CONVOLUTION;
            cna.in_precision = PRECISION_INT8;
            cna.proc_precision = PRECISION_INT8;
            cna.kernel_groups = 0;
            cna.feature_grains = params.m + 1;
            cna.conv_x_stride = 1;
            cna.conv_y_stride = 1;
            cna.datain_width = 1;
            cna.datain_height = params.m;
            cna.datain_channel = params.k;
            cna.dataout_width = 1;
            cna.dataout_height = params.m;
            cna.dataout_atomics = cna.dataout_width as u32 * cna.dataout_height as u32;
            cna.weight_width = 1;
            cna.weight_height = 1;
            cna.weight_kernels = params.n;
            cna.weight_bytes_per_kernel =
                cna.weight_width as u32 * cna.weight_height as u32 * cna.datain_channel as u32;
            cna.weight_bytes = cna.weight_bytes_per_kernel * cna.weight_kernels as u32;

            let fd_bytes =
                cna.datain_width as u32 * cna.datain_height as u32 * cna.datain_channel as u32;
            let fd_banks = div_ceil_u32(fd_bytes, NPU_CBUF_BANK_SIZE);
            let weight_banks_needed = div_ceil_u32(cna.weight_bytes, NPU_CBUF_BANK_SIZE);

            if fd_banks > (NPU_CBUF_BANKS - 1) {
                return Err(MatmulError::CbufOverflow);
            }
            if cna.weight_bytes_per_kernel > NPU_CBUF_BANK_SIZE {
                return Err(MatmulError::KernelTooLarge);
            }
            if weight_banks_needed > (NPU_CBUF_BANKS - fd_banks) {
                return Err(MatmulError::CbufOverflow);
            }
            let weight_banks = NPU_CBUF_BANKS - fd_banks;

            cna.weight_bank = weight_banks as u8;
            cna.data_bank = fd_banks as u8;
            let data_entries =
                div_ceil_u32(cna.datain_width as u32 * cna.datain_channel as u32, 64);
            cna.data_entries = data_entries as u16;
            cna.data_sign = 1;
            cna.cvt_type = 1;
            cna.cvt_bypass = 1;
            cna.cvt_scale0 = 1;
            cna.cvt_scale1 = 1;
            cna.cvt_scale2 = 1;
            cna.cvt_scale3 = 1;
            cna.fc_skip_en = 0;
            cna.data_offset = 0;
            cna.pad_left = 0;
            cna.pad_top = 0;
            cna.feature_base_addr = params.input_dma;
            cna.weight_offset = 0;
            cna.weight_burst_len = 0xF;
            cna.data_burst_len = 0xF;
            cna.line_stride = cna.datain_width as u32 * 4;
            let mut surf_stride = cna.line_stride as i32 * ((cna.datain_height as i32 / 4) - 1);
            if surf_stride < 0 {
                surf_stride += 1;
            }
            cna.surf_stride = surf_stride;
            cna.dma_width = cna.datain_width;
            cna.dma_height = cna.datain_height;
            cna.dma_channel = cna.datain_channel;
            cna.decompress_addr0 = params.weights_dma;

            let core = CoreDesc {
                proc_precision: PRECISION_INT8,
                qd_en: 0,
                dataout_height: cna.dataout_height - 1,
                dataout_width: cna.dataout_width - 1,
                dataout_channel: cna.weight_kernels - 1,
            };

            let mut dpu = DpuDesc::default();
            dpu.burst_len = 0xF;
            dpu.conv_mode = DIRECT_CONVOLUTION;
            dpu.output_mode = 0x2;
            dpu.flying_mode = 0x0;
            dpu.out_precision = PRECISION_INT32;
            dpu.in_precision = PRECISION_INT8;
            dpu.proc_precision = PRECISION_INT8;
            dpu.dst_base_addr = params.output_dma;
            dpu.dst_surf_stride = cna.dataout_height as u32 * cna.dataout_width as u32;
            dpu.width = core.dataout_width;
            dpu.height = core.dataout_height;
            dpu.channel = core.dataout_channel;
            dpu.bs_bypass = 1;
            dpu.bs_alu_bypass = 1;
            dpu.bs_mul_bypass = 1;
            dpu.bs_relu_bypass = 1;
            dpu.bn_bypass = 1;
            dpu.bn_alu_bypass = 1;
            dpu.bn_mul_bypass = 1;
            dpu.bn_relu_bypass = 1;
            dpu.ew_bypass = 1;
            dpu.ew_op_bypass = 1;
            dpu.ew_lut_bypass = 1;
            dpu.ew_op_cvt_bypass = 1;
            dpu.ew_relu_bypass = 1;
            dpu.fp32tofp16_en = 0;
            dpu.out_cvt_scale = 1;
            dpu.size_e_2 = 7;
            dpu.size_e_1 = 7;
            dpu.size_e_0 = 7;
            dpu.od_bypass = 1;
            dpu.channel_wdma = core.dataout_channel;
            dpu.height_wdma = core.dataout_height;
            dpu.width_wdma = core.dataout_width;
            dpu.surf_add = dpu.dst_surf_stride * 8;

            let mut ops = [0u64; REGCMD_WORDS];
            gen_matmul_task(&mut ops, &cna, &core, &dpu);
            Ok(ops)
        }

        fn gen_matmul_task(
            ops: &mut [u64; REGCMD_WORDS],
            cna: &CnaDesc,
            core: &CoreDesc,
            dpu: &DpuDesc,
        ) {
            const PC_OPERATION_ENABLE: u16 = 0x0008;
            const PC_REGISTER_AMOUNTS: u16 = 0x0014;
            const CNA_CONV_CON1: u16 = 0x100C;
            const CNA_CONV_CON2: u16 = 0x1010;
            const CNA_CONV_CON3: u16 = 0x1014;
            const CNA_DATA_SIZE0: u16 = 0x1020;
            const CNA_DATA_SIZE1: u16 = 0x1024;
            const CNA_DATA_SIZE2: u16 = 0x1028;
            const CNA_DATA_SIZE3: u16 = 0x102C;
            const CNA_WEIGHT_SIZE0: u16 = 0x1030;
            const CNA_WEIGHT_SIZE1: u16 = 0x1034;
            const CNA_WEIGHT_SIZE2: u16 = 0x1038;
            const CNA_CBUF_CON0: u16 = 0x1040;
            const CNA_CBUF_CON1: u16 = 0x1044;
            const CNA_CVT_CON0: u16 = 0x104C;
            const CNA_CVT_CON1: u16 = 0x1050;
            const CNA_CVT_CON2: u16 = 0x1054;
            const CNA_CVT_CON3: u16 = 0x1058;
            const CNA_CVT_CON4: u16 = 0x105C;
            const CNA_FC_CON0: u16 = 0x1060;
            const CNA_FC_CON1: u16 = 0x1064;
            const CNA_PAD_CON0: u16 = 0x1068;
            const CNA_FEATURE_DATA_ADDR: u16 = 0x1070;
            const CNA_FC_CON2: u16 = 0x1074;
            const CNA_DMA_CON0: u16 = 0x1078;
            const CNA_DMA_CON1: u16 = 0x107C;
            const CNA_DMA_CON2: u16 = 0x1080;
            const CNA_FC_DATA_SIZE0: u16 = 0x1084;
            const CNA_FC_DATA_SIZE1: u16 = 0x1088;
            const CNA_DCOMP_CTRL: u16 = 0x1100;
            const CNA_DCOMP_REGNUM: u16 = 0x1104;
            const CNA_DCOMP_ADDR0: u16 = 0x1110;
            const CNA_DCOMP_AMOUNT: u16 = 0x1140;
            const CNA_CVT_CON5: u16 = 0x1180;
            const CNA_PAD_CON1: u16 = 0x1184;
            const CORE_MISC_CFG: u16 = 0x3010;
            const CORE_DATAOUT_SIZE_0: u16 = 0x3014;
            const CORE_DATAOUT_SIZE_1: u16 = 0x3018;
            const CORE_CLIP_TRUNCATE: u16 = 0x301C;
            const CORE_3030: u16 = 0x3030;
            const DPU_S_POINTER: u16 = 0x4004;
            const DPU_FEATURE_MODE_CFG: u16 = 0x400C;
            const DPU_DATA_FORMAT: u16 = 0x4010;
            const DPU_OFFSET_PEND: u16 = 0x4014;
            const DPU_DST_BASE_ADD: u16 = 0x4020;
            const DPU_DST_SURF_STRIDE: u16 = 0x4024;
            const DPU_DATA_CUBE_WIDTH: u16 = 0x4030;
            const DPU_DATA_CUBE_HEIGHT: u16 = 0x4034;
            const DPU_DATA_CUBE_NOTCH_ADDR: u16 = 0x4038;
            const DPU_DATA_CUBE_CHANNEL: u16 = 0x403C;
            const DPU_BS_CFG: u16 = 0x4040;
            const DPU_BS_ALU_CFG: u16 = 0x4044;
            const DPU_BS_MUL_CFG: u16 = 0x4048;
            const DPU_BS_RELUX_CMP_VALUE: u16 = 0x404C;
            const DPU_BS_OW_CFG: u16 = 0x4050;
            const DPU_BS_OW_OP: u16 = 0x4054;
            const DPU_WDMA_SIZE_0: u16 = 0x4058;
            const DPU_WDMA_SIZE_1: u16 = 0x405C;
            const DPU_BN_CFG: u16 = 0x4060;
            const DPU_BN_ALU_CFG: u16 = 0x4064;
            const DPU_BN_MUL_CFG: u16 = 0x4068;
            const DPU_BN_RELUX_CMP_VALUE: u16 = 0x406C;
            const DPU_EW_CFG: u16 = 0x4070;
            const DPU_EW_CVT_OFFSET_VALUE: u16 = 0x4074;
            const DPU_EW_CVT_SCALE_VALUE: u16 = 0x4078;
            const DPU_EW_RELUX_CMP_VALUE: u16 = 0x407C;
            const DPU_OUT_CVT_OFFSET: u16 = 0x4080;
            const DPU_OUT_CVT_SCALE: u16 = 0x4084;
            const DPU_OUT_CVT_SHIFT: u16 = 0x4088;
            const DPU_EW_OP_VALUE_0: u16 = 0x4090;
            const DPU_SURFACE_ADD: u16 = 0x40C0;
            const DPU_40C4: u16 = 0x40C4;
            const DPU_LUT_ACCESS_CFG: u16 = 0x4100;
            const DPU_LUT_ACCESS_DATA: u16 = 0x4104;
            const DPU_LUT_CFG: u16 = 0x4108;
            const DPU_LUT_INFO: u16 = 0x410C;
            const DPU_LUT_LE_START: u16 = 0x4110;
            const DPU_LUT_LE_END: u16 = 0x4114;
            const DPU_LUT_LO_START: u16 = 0x4118;
            const DPU_LUT_LO_END: u16 = 0x411C;
            const DPU_LUT_LE_SLOPE_SCALE: u16 = 0x4120;
            const DPU_LUT_LE_SLOPE_SHIFT: u16 = 0x4124;
            const DPU_LUT_LO_SLOPE_SCALE: u16 = 0x4128;
            const DPU_LUT_LO_SLOPE_SHIFT: u16 = 0x412C;

            const BLOCK_PC: u16 = 0x0100;
            const BLOCK_CNA: u16 = 0x0200;
            const BLOCK_CORE: u16 = 0x0800;
            const BLOCK_DPU: u16 = 0x1000;
            const PC_OP_01: u16 = 0x01;
            const PC_OP_40: u16 = 0x40;
            const PC_OP_ENABLE: u16 = 0x80;

            const OP_REG_PC: u16 = BLOCK_PC | PC_OP_01;
            const OP_REG_CNA: u16 = BLOCK_CNA | PC_OP_01;
            const OP_REG_CORE: u16 = BLOCK_CORE | PC_OP_01;
            const OP_REG_DPU: u16 = BLOCK_DPU | PC_OP_01;
            const OP_40: u16 = PC_OP_40 | PC_OP_01;
            const OP_ENABLE: u16 = PC_OP_ENABLE | PC_OP_01;

            const PC_ENABLE: u32 = 0x01;
            const PC_ENABLE_CNA: u32 = 0x04;
            const PC_ENABLE_DPU: u32 = 0x08;

            let mut value;

            ops[0] = npu_op(OP_REG_DPU, 0xE, DPU_S_POINTER);
            value = ((cna.proc_precision as u32 & 0x7) << 7)
                | ((cna.in_precision as u32 & 0x7) << 4)
                | (cna.conv_mode as u32 & 0xF);
            ops[1] = npu_op(OP_REG_CNA, value, CNA_CONV_CON1);
            value = ((cna.kernel_groups as u32 & 0xFF) << 16)
                | ((cna.feature_grains as u32 & 0x3FF) << 4);
            ops[2] = npu_op(OP_REG_CNA, value, CNA_CONV_CON2);
            value = ((cna.conv_y_stride as u32 & 0x7) << 3) | (cna.conv_x_stride as u32 & 0x7);
            ops[3] = npu_op(OP_REG_CNA, value, CNA_CONV_CON3);
            value = ((cna.datain_width as u32 & 0x7FF) << 16) | (cna.datain_height as u32 & 0x7FF);
            ops[4] = npu_op(OP_REG_CNA, value, CNA_DATA_SIZE0);
            value = ((cna.datain_channel as u32 - 1) & 0xFFFF) << 16
                | (cna.datain_channel as u32 & 0xFFFF);
            ops[5] = npu_op(OP_REG_CNA, value, CNA_DATA_SIZE1);
            value = cna.dataout_width as u32 & 0x7FF;
            ops[6] = npu_op(OP_REG_CNA, value, CNA_DATA_SIZE2);
            value = cna.dataout_atomics & 0x3FFFF;
            ops[7] = npu_op(OP_REG_CNA, value, CNA_DATA_SIZE3);
            ops[8] = npu_op(OP_REG_CNA, cna.weight_bytes, CNA_WEIGHT_SIZE0);
            ops[9] = npu_op(
                OP_REG_CNA,
                cna.weight_bytes_per_kernel & 0x7FFFF,
                CNA_WEIGHT_SIZE1,
            );
            value = ((cna.weight_width as u32 & 0x1F) << 24)
                | ((cna.weight_height as u32 & 0x1F) << 16)
                | (cna.weight_kernels as u32 & 0x3FFF);
            ops[10] = npu_op(OP_REG_CNA, value, CNA_WEIGHT_SIZE2);
            value = ((cna.weight_bank as u32 & 0xF) << 4) | (cna.data_bank as u32 & 0xF);
            ops[11] = npu_op(OP_REG_CNA, value, CNA_CBUF_CON0);
            ops[12] = npu_op(OP_REG_CNA, cna.data_entries as u32 & 0x1FFF, CNA_CBUF_CON1);
            value = ((cna.data_sign as u32 & 0x1) << 3)
                | ((cna.cvt_type as u32 & 0x1) << 1)
                | (cna.cvt_bypass as u32 & 0x1);
            ops[13] = npu_op(OP_REG_CNA, value, CNA_CVT_CON0);
            value = (cna.cvt_scale0 as u32 & 0xFFFF) << 16;
            ops[14] = npu_op(OP_REG_CNA, value, CNA_CVT_CON1);
            value = (cna.cvt_scale1 as u32 & 0xFFFF) << 16;
            ops[15] = npu_op(OP_REG_CNA, value, CNA_CVT_CON2);
            value = (cna.cvt_scale2 as u32 & 0xFFFF) << 16;
            ops[16] = npu_op(OP_REG_CNA, value, CNA_CVT_CON3);
            value = (cna.cvt_scale3 as u32 & 0xFFFF) << 16;
            ops[17] = npu_op(OP_REG_CNA, value, CNA_CVT_CON4);
            ops[18] = npu_op(OP_REG_CNA, cna.fc_skip_en as u32 & 0x1, CNA_FC_CON0);
            ops[19] = npu_op(OP_REG_CNA, cna.data_offset as u32 & 0x1FFFF, CNA_FC_CON1);
            value = ((cna.pad_left as u32 & 0xF) << 4) | (cna.pad_top as u32 & 0xF);
            ops[20] = npu_op(OP_REG_CNA, value, CNA_PAD_CON0);
            ops[21] = npu_op(OP_REG_CNA, cna.feature_base_addr, CNA_FEATURE_DATA_ADDR);
            ops[22] = npu_op(OP_REG_CNA, cna.weight_offset as u32 & 0x1FFFF, CNA_FC_CON2);
            value = ((cna.weight_burst_len as u32 & 0xF) << 16) | (cna.data_burst_len as u32 & 0xF);
            ops[23] = npu_op(OP_REG_CNA, value, CNA_DMA_CON0);
            ops[24] = npu_op(OP_REG_CNA, cna.line_stride & 0x0FFF_FFFF, CNA_DMA_CON1);
            ops[25] = npu_op(
                OP_REG_CNA,
                (cna.surf_stride as u32) & 0x0FFF_FFFF,
                CNA_DMA_CON2,
            );
            value = ((cna.dma_width as u32 & 0x7FF) << 16) | (cna.dma_height as u32 & 0x7FF);
            ops[26] = npu_op(OP_REG_CNA, value, CNA_FC_DATA_SIZE0);
            ops[27] = npu_op(
                OP_REG_CNA,
                cna.dma_channel as u32 & 0xFFFF,
                CNA_FC_DATA_SIZE1,
            );
            ops[28] = npu_op(OP_REG_CNA, 0, CNA_DCOMP_CTRL);
            ops[29] = npu_op(OP_REG_CNA, 0, CNA_DCOMP_REGNUM);
            ops[30] = npu_op(OP_REG_CNA, cna.decompress_addr0, CNA_DCOMP_ADDR0);
            for (offset, slot) in ops[31..47].iter_mut().enumerate() {
                *slot = npu_op(OP_REG_CNA, 0, CNA_DCOMP_AMOUNT + (offset as u16 * 4));
            }
            ops[47] = npu_op(OP_REG_CNA, 0, CNA_CVT_CON5);
            ops[48] = npu_op(OP_REG_CNA, 0, CNA_PAD_CON1);

            value = ((core.proc_precision as u32 & 0x7) << 8) | (core.qd_en as u32 & 0x1);
            ops[49] = npu_op(OP_REG_CORE, value, CORE_MISC_CFG);
            value = ((core.dataout_height as u32 & 0xFFFF) << 16)
                | (core.dataout_width as u32 & 0xFFFF);
            ops[50] = npu_op(OP_REG_CORE, value, CORE_DATAOUT_SIZE_0);
            ops[51] = npu_op(
                OP_REG_CORE,
                core.dataout_channel as u32 & 0xFFFF,
                CORE_DATAOUT_SIZE_1,
            );
            ops[52] = npu_op(OP_REG_CORE, 0, CORE_CLIP_TRUNCATE);
            ops[53] = npu_op(OP_REG_CORE, 0, CORE_3030);

            value = ((dpu.burst_len as u32 & 0xF) << 5)
                | ((dpu.conv_mode as u32 & 0x3) << 3)
                | ((dpu.output_mode as u32 & 0x3) << 1)
                | (dpu.flying_mode as u32 & 0x1);
            ops[54] = npu_op(OP_REG_DPU, value, DPU_FEATURE_MODE_CFG);
            value = ((dpu.out_precision as u32 & 0x7) << 29)
                | ((dpu.in_precision as u32 & 0x7) << 26)
                | (dpu.proc_precision as u32 & 0x7);
            ops[55] = npu_op(OP_REG_DPU, value, DPU_DATA_FORMAT);
            ops[56] = npu_op(OP_REG_DPU, 0, DPU_OFFSET_PEND);
            ops[57] = npu_op(OP_REG_DPU, dpu.dst_base_addr, DPU_DST_BASE_ADD);
            ops[58] = npu_op(
                OP_REG_DPU,
                (dpu.dst_surf_stride & 0x0FFF_FFFF) << 4,
                DPU_DST_SURF_STRIDE,
            );
            ops[59] = npu_op(OP_REG_DPU, dpu.width as u32 & 0x1FFF, DPU_DATA_CUBE_WIDTH);
            ops[60] = npu_op(OP_REG_DPU, dpu.height as u32 & 0x1FFF, DPU_DATA_CUBE_HEIGHT);
            ops[61] = npu_op(OP_REG_DPU, 0, DPU_DATA_CUBE_NOTCH_ADDR);
            value = ((dpu.channel as u32 & 0x1FFF) << 16) | (dpu.channel as u32 & 0x1FFF);
            ops[62] = npu_op(OP_REG_DPU, value, DPU_DATA_CUBE_CHANNEL);
            value = ((dpu.bs_relu_bypass as u32 & 0x1) << 6)
                | ((dpu.bs_mul_bypass as u32 & 0x1) << 4)
                | ((dpu.bs_alu_bypass as u32 & 0x1) << 1)
                | (dpu.bs_bypass as u32 & 0x1);
            ops[63] = npu_op(OP_REG_DPU, value, DPU_BS_CFG);
            ops[64] = npu_op(OP_REG_DPU, 0, DPU_BS_ALU_CFG);
            ops[65] = npu_op(OP_REG_DPU, 0, DPU_BS_MUL_CFG);
            ops[66] = npu_op(OP_REG_DPU, 0, DPU_BS_RELUX_CMP_VALUE);
            value = ((dpu.size_e_2 as u32 & 0x7) << 8)
                | ((dpu.size_e_1 as u32 & 0x7) << 5)
                | ((dpu.size_e_0 as u32 & 0x7) << 2)
                | ((dpu.od_bypass as u32 & 0x1) << 1);
            ops[67] = npu_op(OP_REG_DPU, value, DPU_BS_OW_CFG);
            ops[68] = npu_op(OP_REG_DPU, 0, DPU_BS_OW_OP);
            ops[69] = npu_op(
                OP_REG_DPU,
                dpu.channel_wdma as u32 & 0x1FFF,
                DPU_WDMA_SIZE_0,
            );
            value = ((dpu.height_wdma as u32 & 0x1FFF) << 16) | (dpu.width_wdma as u32 & 0x1FFF);
            ops[70] = npu_op(OP_REG_DPU, value, DPU_WDMA_SIZE_1);
            value = ((dpu.bn_relu_bypass as u32 & 0x1) << 6)
                | ((dpu.bn_mul_bypass as u32 & 0x1) << 4)
                | ((dpu.bn_alu_bypass as u32 & 0x1) << 1)
                | (dpu.bn_bypass as u32 & 0x1);
            ops[71] = npu_op(OP_REG_DPU, value, DPU_BN_CFG);
            ops[72] = npu_op(OP_REG_DPU, 0, DPU_BN_ALU_CFG);
            ops[73] = npu_op(OP_REG_DPU, 0, DPU_BN_MUL_CFG);
            ops[74] = npu_op(OP_REG_DPU, 0, DPU_BN_RELUX_CMP_VALUE);
            value = ((dpu.ew_relu_bypass as u32 & 0x1) << 9)
                | ((dpu.ew_op_cvt_bypass as u32 & 0x1) << 8)
                | ((dpu.ew_lut_bypass as u32 & 0x1) << 7)
                | ((dpu.ew_op_bypass as u32 & 0x1) << 1)
                | (dpu.ew_bypass as u32 & 0x1);
            ops[75] = npu_op(OP_REG_DPU, value, DPU_EW_CFG);
            ops[76] = npu_op(OP_REG_DPU, 0, DPU_EW_CVT_OFFSET_VALUE);
            ops[77] = npu_op(OP_REG_DPU, 0x1, DPU_EW_CVT_SCALE_VALUE);
            ops[78] = npu_op(OP_REG_DPU, 0, DPU_EW_RELUX_CMP_VALUE);
            ops[79] = npu_op(OP_REG_DPU, 0, DPU_OUT_CVT_OFFSET);
            value = ((dpu.fp32tofp16_en as u32 & 0x1) << 16) | (dpu.out_cvt_scale as u32 & 0xFFFF);
            ops[80] = npu_op(OP_REG_DPU, value, DPU_OUT_CVT_SCALE);
            ops[81] = npu_op(OP_REG_DPU, 0, DPU_OUT_CVT_SHIFT);
            for i in 0..8 {
                ops[82 + i] = npu_op(OP_REG_DPU, 0, DPU_EW_OP_VALUE_0 + (i as u16 * 4));
            }
            ops[90] = npu_op(
                OP_REG_DPU,
                (dpu.surf_add & 0x0FFF_FFFF) << 4,
                DPU_SURFACE_ADD,
            );
            ops[91] = npu_op(OP_REG_DPU, 0, DPU_40C4);
            ops[92] = npu_op(OP_REG_DPU, 0, DPU_LUT_ACCESS_CFG);
            ops[93] = npu_op(OP_REG_DPU, 0, DPU_LUT_ACCESS_DATA);
            ops[94] = npu_op(OP_REG_DPU, 0, DPU_LUT_CFG);
            ops[95] = npu_op(OP_REG_DPU, 0, DPU_LUT_INFO);
            ops[96] = npu_op(OP_REG_DPU, 0, DPU_LUT_LE_START);
            ops[97] = npu_op(OP_REG_DPU, 0, DPU_LUT_LE_END);
            ops[98] = npu_op(OP_REG_DPU, 0, DPU_LUT_LO_START);
            ops[99] = npu_op(OP_REG_DPU, 0, DPU_LUT_LO_END);
            ops[100] = npu_op(OP_REG_DPU, 0, DPU_LUT_LE_SLOPE_SCALE);
            ops[101] = npu_op(OP_REG_DPU, 0, DPU_LUT_LE_SLOPE_SHIFT);
            ops[102] = npu_op(OP_REG_DPU, 0, DPU_LUT_LO_SLOPE_SCALE);
            ops[103] = npu_op(OP_REG_DPU, 0, DPU_LUT_LO_SLOPE_SHIFT);
            ops[104] = npu_op(0, 0, 0);
            ops[105] = npu_op(OP_REG_PC, 0, PC_REGISTER_AMOUNTS);
            ops[106] = npu_op(OP_40, 0, 0);
            ops[107] = npu_op(
                OP_ENABLE,
                PC_ENABLE | PC_ENABLE_CNA | PC_ENABLE_DPU,
                PC_OPERATION_ENABLE,
            );
        }

        fn npu_op(op: u16, value: u32, reg: u16) -> u64 {
            ((op as u64 & 0xFFFF) << 48) | ((value as u64 & 0xFFFF_FFFF) << 16) | reg as u64
        }

        fn feature_data(
            _channels: usize,
            h: usize,
            w: usize,
            c2: usize,
            chan: usize,
            row: usize,
            col: usize,
        ) -> usize {
            let plane = (chan - 1) / c2;
            let src = plane * h * w * c2;
            let offset = (chan - 1) % c2;
            src + c2 * ((row - 1) * w + (col - 1)) + offset
        }

        fn weight_int8(c: usize, k: usize, chan: usize) -> usize {
            let kpg = (k - 1) / 32;
            let cpg = (chan - 1) / 32;
            let mut dst = (cpg * 32) * 32 + (kpg * 32 * c);
            dst += (chan - 1) % 32;
            dst += ((k - 1) % 32) * 32;
            dst
        }
    }
}
