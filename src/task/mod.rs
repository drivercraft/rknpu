use alloc::vec::Vec;
use dma_api::{DVec, Direction};

use crate::{
    JobMode, RKNPU_PC_DATA_EXTRA_AMOUNT, RknpuTask,
    cna::{NpuCnaDesc, NpuCoreDesc},
    dpu::NpuDpuDesc,
    op::Operation,
};

pub mod cna;
mod def;
pub mod dpu;
pub mod op;

use def::*;

const DIRECT_CONVOLUTION: u8 = 0x0;
const NPU_CBUF_BANK_SIZE: u16 = 32768;
const NPU_CBUF_BANKS: u16 = 12;

#[allow(unused)]
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
enum Precision {
    Int8 = 0x0,
    Float16 = 0x2,
    Int32 = 0x4,
    Float32 = 0x5,
}

pub struct Sharp {
    pub width: usize,
    pub height: usize,
}

/// u8 matrix multiplication operation
pub struct Matmul {
    m: u16,
    k: u16,
    n: u16,
    reg_cmds: DVec<u64>,
    input: DVec<i8>,
    weight: DVec<i8>,
    output: DVec<i32>,
}

impl Matmul {
    pub fn new(a: Sharp, b: Sharp) -> Self {
        assert_eq!(a.width, b.height);
        let m = a.height;
        let k = a.width;
        let n = b.width;

        Self {
            m: m as _,
            k: k as _,
            n: n as _,
            reg_cmds: DVec::zeros(u32::MAX as _, 112, 0x1000, Direction::Bidirectional).unwrap(),
            input: DVec::zeros(
                u32::MAX as _,
                m * k * size_of::<u8>(),
                0x1000,
                Direction::Bidirectional,
            )
            .unwrap(),
            weight: DVec::zeros(
                u32::MAX as _,
                k * n * size_of::<u8>(),
                0x1000,
                Direction::Bidirectional,
            )
            .unwrap(),
            output: DVec::zeros(u32::MAX as _, m * n, 0x1000, Direction::Bidirectional).unwrap(),
        }
    }

    pub fn output(&self) -> &[i32] {
        self.output.as_ref()
    }

    fn as_task(&mut self) -> RknpuTask {
        let mut cna_desc = NpuCnaDesc::default();
        let mut core_desc = NpuCoreDesc::default();
        let mut dpu_desc = NpuDpuDesc::default();

        debug!(
            "Generating matmul task: M={}, K={}, N={}",
            self.m, self.k, self.n
        );
        debug!("Input feature address: {:#x}", self.input.bus_addr());
        debug!("Weight address: {:#x}", self.weight.bus_addr());
        debug!("Output address: {:#x}", self.output.bus_addr());

        cna_desc.conv_mode = DIRECT_CONVOLUTION;
        cna_desc.in_precision = Precision::Int8 as u8;
        cna_desc.proc_precision = Precision::Int8 as u8;

        cna_desc.kernel_groups = 0;
        cna_desc.feature_grains = self.m + 1;
        cna_desc.conv_x_stride = 1;
        cna_desc.conv_y_stride = 1;

        cna_desc.datain_width = 1;
        cna_desc.datain_height = self.m;
        cna_desc.datain_channel = self.k;
        cna_desc.dataout_width = 1;
        cna_desc.dataout_height = self.m;
        cna_desc.dataout_atomics = cna_desc.dataout_width as u32 * cna_desc.dataout_height as u32;

        cna_desc.weight_width = 1;
        cna_desc.weight_height = 1;
        cna_desc.weight_kernels = self.n;
        cna_desc.weight_bytes_per_kernel = cna_desc.weight_width as u32
            * cna_desc.weight_height as u32
            * cna_desc.datain_channel as u32
            * size_of::<u8>() as u32;
        cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels as u32;

        let fd_bytes = cna_desc.datain_width
            * cna_desc.datain_height
            * cna_desc.datain_channel
            * size_of::<u8>() as u16;
        let mut fd_banks = fd_bytes / NPU_CBUF_BANK_SIZE;
        fd_banks = if fd_bytes.is_multiple_of(NPU_CBUF_BANK_SIZE) {
            fd_banks
        } else {
            fd_banks + 1
        };
        let mut weight_banks = cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE as u32;
        weight_banks = if (cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE as u32) == 0 {
            weight_banks
        } else {
            weight_banks + 1
        };
        if (fd_banks) > NPU_CBUF_BANKS - 1 {
            panic!("Input feature data size exceed cbuf size");
        } else if cna_desc.weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE as u32 {
            weight_banks = NPU_CBUF_BANKS as u32 - fd_banks as u32;
        } else {
            panic!("Weight data size exceed cbuf size");
        }

        cna_desc.weight_bank = weight_banks as _;
        cna_desc.data_bank = fd_banks as _;
        cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) / 64;
        cna_desc.data_entries = if (cna_desc.datain_width * cna_desc.datain_channel) % 64 == 0 {
            cna_desc.data_entries
        } else {
            cna_desc.data_entries + 1
        };
        cna_desc.data_sign = 0x1;
        cna_desc.cvt_type = 0x1;
        cna_desc.cvt_bypass = 0x1;
        cna_desc.cvt_scale0 = 0x1;
        cna_desc.cvt_scale1 = 0x1;
        cna_desc.cvt_scale2 = 0x1;
        cna_desc.cvt_scale3 = 0x1;
        cna_desc.fc_skip_en = 0;
        cna_desc.data_offset = 0x0;
        cna_desc.pad_left = 0;
        cna_desc.pad_top = 0;
        cna_desc.feature_base_addr = self.input.bus_addr() as u32;
        cna_desc.weight_offset = 0;
        cna_desc.weight_burst_len = 0xf;
        cna_desc.data_burst_len = 0xf;
        cna_desc.line_stride = cna_desc.datain_width as u32 * 4;
        let mut surf_stride =
            cna_desc.line_stride as i32 * ((cna_desc.datain_height as i32 / 4) - 1);
        surf_stride = if surf_stride < 0 {
            surf_stride + 1
        } else {
            surf_stride
        };
        cna_desc.surf_stride = surf_stride;
        cna_desc.dma_width = cna_desc.datain_width;
        cna_desc.dma_height = cna_desc.datain_height;
        cna_desc.dma_channel = cna_desc.datain_channel;
        cna_desc.decompress_addr0 = self.weight.bus_addr() as _;

        core_desc.proc_precision = Precision::Int8 as u8;
        core_desc.qd_en = 0;
        core_desc.dataout_height = cna_desc.dataout_height - 1;
        core_desc.dataout_width = cna_desc.dataout_width - 1;
        core_desc.dataout_channel = cna_desc.weight_kernels - 1;

        dpu_desc.burst_len = 0xf;
        dpu_desc.conv_mode = DIRECT_CONVOLUTION;
        dpu_desc.output_mode = 0x2;
        dpu_desc.flying_mode = 0x0;
        dpu_desc.out_precision = Precision::Int32 as u8;
        dpu_desc.in_precision = Precision::Int8 as u8;
        dpu_desc.proc_precision = Precision::Int8 as u8;
        dpu_desc.dst_base_addr = self.output.bus_addr() as _;
        dpu_desc.dst_surf_stride = cna_desc.dataout_height as u32 * cna_desc.dataout_width as u32;
        dpu_desc.width = core_desc.dataout_width;
        dpu_desc.height = core_desc.dataout_height;
        dpu_desc.channel = core_desc.dataout_channel;
        dpu_desc.bs_bypass = 1;
        dpu_desc.bs_alu_bypass = 1;
        dpu_desc.bs_mul_bypass = 1;
        dpu_desc.bs_relu_bypass = 1;
        dpu_desc.bn_bypass = 1;
        dpu_desc.bn_alu_bypass = 1;
        dpu_desc.bn_mul_bypass = 1;
        dpu_desc.bn_relu_bypass = 1;
        dpu_desc.ew_bypass = 1;
        dpu_desc.ew_op_bypass = 1;
        dpu_desc.ew_lut_bypass = 1;
        dpu_desc.ew_op_cvt_bypass = 1;
        dpu_desc.ew_relu_bypass = 1;
        dpu_desc.fp32tofp16_en = 0;
        dpu_desc.out_cvt_scale = 1;
        dpu_desc.size_e_2 = 7;
        dpu_desc.size_e_1 = 7;
        dpu_desc.size_e_0 = 7;
        dpu_desc.od_bypass = 1;
        dpu_desc.width_wdma = core_desc.dataout_width;
        dpu_desc.height_wdma = core_desc.dataout_height;
        dpu_desc.channel_wdma = core_desc.dataout_channel;
        dpu_desc.surf_add = dpu_desc.dst_surf_stride * 8;

        self.gen_matul(&cna_desc, &core_desc, &dpu_desc);

        // Prepare register commands
        RknpuTask {
            flags: 0,
            op_idx: 0,
            enable_mask: 0xd,
            int_mask: 0x300, // wait for DPU to finish
            int_clear: 0x1ffff,
            int_status: 0,
            regcfg_amount: self.reg_cmds.len() as u32 - (RKNPU_PC_DATA_EXTRA_AMOUNT + 4),
            regcfg_offset: 0,
            regcmd_addr: self.reg_cmds.bus_addr(),
        }
    }

    fn gen_matul(&mut self, cna: &NpuCnaDesc, core: &NpuCoreDesc, dpu: &NpuDpuDesc) {
        // Generate register commands for the matmul operation
        let mut value = 0;

        self.reg_cmds.set(
            0,
            npu_op(OP_REG_DPU, value, DPU_S_POINTER), // CNA_DESC_BASE_ADDR
        );

        value = ((cna.proc_precision as u32 & 0x7) << 7)
            | ((cna.in_precision as u32 & 0x7) << 4)
            | (cna.conv_mode as u32 & 0xF);
        self.reg_cmds
            .set(1, npu_op(OP_REG_CNA, value, CNA_CONV_CON1));
        value =
            ((cna.kernel_groups as u32 & 0xFF) << 16) | ((cna.feature_grains as u32 & 0x3FF) << 4);
        self.reg_cmds
            .set(2, npu_op(OP_REG_CNA, value, CNA_CONV_CON2));
        value = ((cna.conv_y_stride as u32 & 0x7) << 3) | (cna.conv_x_stride as u32 & 0x7);
        self.reg_cmds
            .set(3, npu_op(OP_REG_CNA, value, CNA_CONV_CON3));
        value = ((cna.datain_width as u32 & 0x7FF) << 16) | (cna.datain_height as u32 & 0x7FF);
        self.reg_cmds
            .set(4, npu_op(OP_REG_CNA, value, CNA_DATA_SIZE0));
        value = (((cna.datain_channel - 1) as u32 & 0xFFFF) << 16)
            | (cna.datain_channel as u32 & 0xFFFF);
        self.reg_cmds
            .set(5, npu_op(OP_REG_CNA, value, CNA_DATA_SIZE1));
        value = cna.dataout_width as u32 & 0x7FF;
        self.reg_cmds
            .set(6, npu_op(OP_REG_CNA, value, CNA_DATA_SIZE2));
        value = cna.dataout_atomics & 0x3FFFF;
        self.reg_cmds
            .set(7, npu_op(OP_REG_CNA, value, CNA_DATA_SIZE3));
        value = cna.weight_bytes;
        self.reg_cmds
            .set(8, npu_op(OP_REG_CNA, value, CNA_WEIGHT_SIZE0));
        value = cna.weight_bytes_per_kernel & 0x7FFFF;
        self.reg_cmds
            .set(9, npu_op(OP_REG_CNA, value, CNA_WEIGHT_SIZE1));
        value = ((cna.weight_width as u32 & 0x1F) << 24)
            | ((cna.weight_height as u32 & 0x1F) << 16)
            | (cna.weight_kernels as u32 & 0x3FFF);
        self.reg_cmds
            .set(10, npu_op(OP_REG_CNA, value, CNA_WEIGHT_SIZE2));
        value = ((cna.weight_bank as u32 & 0xF) << 4) | (cna.data_bank as u32 & 0xF);
        self.reg_cmds
            .set(11, npu_op(OP_REG_CNA, value, CNA_CBUF_CON0));
        value = cna.data_entries as u32 & 0x1FFF;
        self.reg_cmds
            .set(12, npu_op(OP_REG_CNA, value, CNA_CBUF_CON1));
        value = ((cna.data_sign as u32 & 0x1) << 3)
            | ((cna.cvt_type as u32 & 0x1) << 1)
            | (cna.cvt_bypass as u32 & 0x1);
        self.reg_cmds
            .set(13, npu_op(OP_REG_CNA, value, CNA_CVT_CON0));
        value = (cna.cvt_scale0 as u32 & 0xFFFF) << 16;
        self.reg_cmds
            .set(14, npu_op(OP_REG_CNA, value, CNA_CVT_CON1));
        value = (cna.cvt_scale1 as u32 & 0xFFFF) << 16;
        self.reg_cmds
            .set(15, npu_op(OP_REG_CNA, value, CNA_CVT_CON2));
        value = (cna.cvt_scale2 as u32 & 0xFFFF) << 16;
        self.reg_cmds
            .set(16, npu_op(OP_REG_CNA, value, CNA_CVT_CON3));
        value = (cna.cvt_scale3 as u32 & 0xFFFF) << 16;
        self.reg_cmds
            .set(17, npu_op(OP_REG_CNA, value, CNA_CVT_CON4));
        value = cna.fc_skip_en as u32 & 0x1;
        self.reg_cmds
            .set(18, npu_op(OP_REG_CNA, value, CNA_FC_CON0));
        value = cna.data_offset as u32 & 0x1FFFF;
        self.reg_cmds
            .set(19, npu_op(OP_REG_CNA, value, CNA_FC_CON1));
        value = ((cna.pad_left as u32 & 0xF) << 4) | (cna.pad_top as u32 & 0xF);
        self.reg_cmds
            .set(20, npu_op(OP_REG_CNA, value, CNA_PAD_CON0));
        self.reg_cmds.set(
            21,
            npu_op(OP_REG_CNA, cna.feature_base_addr, CNA_FEATURE_DATA_ADDR),
        );
        value = cna.weight_offset as u32 & 0x1FFFF;
        self.reg_cmds
            .set(22, npu_op(OP_REG_CNA, value, CNA_FC_CON2));
        value = ((cna.weight_burst_len as u32 & 0xF) << 16) | (cna.data_burst_len as u32 & 0xF);
        self.reg_cmds
            .set(23, npu_op(OP_REG_CNA, value, CNA_DMA_CON0));
        value = cna.line_stride & 0xFFFFFFF;
        self.reg_cmds
            .set(24, npu_op(OP_REG_CNA, value, CNA_DMA_CON1));
        value = cna.surf_stride as u32 & 0xFFFFFFF;
        self.reg_cmds
            .set(25, npu_op(OP_REG_CNA, value, CNA_DMA_CON2));
        value = ((cna.dma_width as u32 & 0x7FF) << 16) | (cna.dma_height as u32 & 0x7FF);
        self.reg_cmds
            .set(26, npu_op(OP_REG_CNA, value, CNA_FC_DATA_SIZE0));
        value = cna.dma_channel as u32 & 0xFFFF;
        self.reg_cmds
            .set(27, npu_op(OP_REG_CNA, value, CNA_FC_DATA_SIZE1));
        self.reg_cmds
            .set(28, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_CTRL));
        self.reg_cmds
            .set(29, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_REGNUM));
        self.reg_cmds.set(
            30,
            npu_op(OP_REG_CNA, cna.decompress_addr0, CNA_DCOMP_ADDR0),
        );
        self.reg_cmds
            .set(31, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT));
        self.reg_cmds
            .set(32, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT1));
        self.reg_cmds
            .set(33, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT2));
        self.reg_cmds
            .set(34, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT3));
        self.reg_cmds
            .set(35, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT4));
        self.reg_cmds
            .set(36, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT5));
        self.reg_cmds
            .set(37, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT6));
        self.reg_cmds
            .set(38, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT7));
        self.reg_cmds
            .set(39, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT8));
        self.reg_cmds
            .set(40, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT9));
        self.reg_cmds
            .set(41, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT10));
        self.reg_cmds
            .set(42, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT11));
        self.reg_cmds
            .set(43, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT12));
        self.reg_cmds
            .set(44, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT13));
        self.reg_cmds
            .set(45, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT14));
        self.reg_cmds
            .set(46, npu_op(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT15));
        self.reg_cmds.set(47, npu_op(OP_REG_CNA, 0x0, CNA_CVT_CON5));
        self.reg_cmds.set(48, npu_op(OP_REG_CNA, 0x0, CNA_PAD_CON1));
        value = ((core.proc_precision as u32 & 0x7) << 8) | (core.qd_en as u32 & 0x1);
        self.reg_cmds
            .set(49, npu_op(OP_REG_CORE, value, CORE_MISC_CFG));
        value =
            ((core.dataout_height as u32 & 0xFFFF) << 16) | (core.dataout_width as u32 & 0xFFFF);
        self.reg_cmds
            .set(50, npu_op(OP_REG_CORE, value, CORE_DATAOUT_SIZE_0));
        value = core.dataout_channel as u32 & 0xFFFF;
        self.reg_cmds
            .set(51, npu_op(OP_REG_CORE, value, CORE_DATAOUT_SIZE_1));
        self.reg_cmds
            .set(52, npu_op(OP_REG_CORE, 0x0, CORE_CLIP_TRUNCATE));
        self.reg_cmds.set(53, npu_op(OP_REG_CORE, 0x0, CORE_3030));
        value = ((dpu.burst_len as u32 & 0xF) << 5)
            | ((dpu.conv_mode as u32 & 0x3) << 3)
            | ((dpu.output_mode as u32 & 0x3) << 1)
            | (dpu.flying_mode as u32 & 0x1);
        self.reg_cmds
            .set(54, npu_op(OP_REG_DPU, value, DPU_FEATURE_MODE_CFG));
        value = ((dpu.out_precision as u32 & 0x7) << 29)
            | ((dpu.in_precision as u32 & 0x7) << 26)
            | (dpu.proc_precision as u32 & 0x7);
        self.reg_cmds
            .set(55, npu_op(OP_REG_DPU, value, DPU_DATA_FORMAT));
        self.reg_cmds
            .set(56, npu_op(OP_REG_DPU, 0x0, DPU_OFFSET_PEND));
        self.reg_cmds
            .set(57, npu_op(OP_REG_DPU, dpu.dst_base_addr, DPU_DST_BASE_ADD));
        value = (dpu.dst_surf_stride & 0xFFFFFFF) << 4;
        self.reg_cmds
            .set(58, npu_op(OP_REG_DPU, value, DPU_DST_SURF_STRIDE));
        value = dpu.width as u32 & 0x1FFF;
        self.reg_cmds
            .set(59, npu_op(OP_REG_DPU, value, DPU_DATA_CUBE_WIDTH));
        value = dpu.height as u32 & 0x1FFF;
        self.reg_cmds
            .set(60, npu_op(OP_REG_DPU, value, DPU_DATA_CUBE_HEIGHT));
        self.reg_cmds
            .set(61, npu_op(OP_REG_DPU, 0x0, DPU_DATA_CUBE_NOTCH_ADDR));
        value = ((dpu.channel as u32 & 0x1FFF) << 16) | (dpu.channel as u32 & 0x1FFF);
        self.reg_cmds
            .set(62, npu_op(OP_REG_DPU, value, DPU_DATA_CUBE_CHANNEL));
        value = ((dpu.bs_relu_bypass as u32 & 0x1) << 6)
            | ((dpu.bs_mul_bypass as u32 & 0x1) << 4)
            | ((dpu.bs_alu_bypass as u32 & 0x1) << 1)
            | (dpu.bs_bypass as u32 & 0x1);
        self.reg_cmds.set(63, npu_op(OP_REG_DPU, value, DPU_BS_CFG));
        self.reg_cmds
            .set(64, npu_op(OP_REG_DPU, 0x0, DPU_BS_ALU_CFG));
        self.reg_cmds
            .set(65, npu_op(OP_REG_DPU, 0x0, DPU_BS_MUL_CFG));
        self.reg_cmds
            .set(66, npu_op(OP_REG_DPU, 0x0, DPU_BS_RELUX_CMP_VALUE));
        value = ((dpu.size_e_2 as u32 & 0x7) << 8)
            | ((dpu.size_e_1 as u32 & 0x7) << 5)
            | ((dpu.size_e_0 as u32 & 0x7) << 2)
            | ((dpu.od_bypass as u32 & 0x1) << 1);
        self.reg_cmds
            .set(67, npu_op(OP_REG_DPU, value, DPU_BS_OW_CFG));
        self.reg_cmds.set(68, npu_op(OP_REG_DPU, 0x0, DPU_BS_OW_OP));
        value = dpu.channel_wdma as u32 & 0x1FFF;
        self.reg_cmds
            .set(69, npu_op(OP_REG_DPU, value, DPU_WDMA_SIZE_0));
        value = ((dpu.height_wdma as u32 & 0x1FFF) << 16) | (dpu.width_wdma as u32 & 0x1FFF);
        self.reg_cmds
            .set(70, npu_op(OP_REG_DPU, value, DPU_WDMA_SIZE_1));
        value = ((dpu.bn_relu_bypass as u32 & 0x1) << 6)
            | ((dpu.bn_mul_bypass as u32 & 0x1) << 4)
            | ((dpu.bn_alu_bypass as u32 & 0x1) << 1)
            | (dpu.bn_bypass as u32 & 0x1);
        self.reg_cmds.set(71, npu_op(OP_REG_DPU, value, DPU_BN_CFG));
        self.reg_cmds
            .set(72, npu_op(OP_REG_DPU, 0x0, DPU_BN_ALU_CFG));
        self.reg_cmds
            .set(73, npu_op(OP_REG_DPU, 0x0, DPU_BN_MUL_CFG));
        self.reg_cmds
            .set(74, npu_op(OP_REG_DPU, 0x0, DPU_BN_RELUX_CMP_VALUE));
        value = ((dpu.ew_relu_bypass as u32 & 0x1) << 9)
            | ((dpu.ew_op_cvt_bypass as u32 & 0x1) << 8)
            | ((dpu.ew_lut_bypass as u32 & 0x1) << 7)
            | ((dpu.ew_op_bypass as u32 & 0x1) << 1)
            | (dpu.ew_bypass as u32 & 0x1);
        self.reg_cmds.set(75, npu_op(OP_REG_DPU, value, DPU_EW_CFG));
        self.reg_cmds
            .set(76, npu_op(OP_REG_DPU, 0x0, DPU_EW_CVT_OFFSET_VALUE));
        self.reg_cmds
            .set(77, npu_op(OP_REG_DPU, 0x1, DPU_EW_CVT_SCALE_VALUE));
        self.reg_cmds
            .set(78, npu_op(OP_REG_DPU, 0x0, DPU_EW_RELUX_CMP_VALUE));
        self.reg_cmds
            .set(79, npu_op(OP_REG_DPU, 0x0, DPU_OUT_CVT_OFFSET));
        value = ((dpu.fp32tofp16_en as u32 & 0x1) << 16) | (dpu.out_cvt_scale as u32 & 0xFFFF);
        self.reg_cmds
            .set(80, npu_op(OP_REG_DPU, value, DPU_OUT_CVT_SCALE));
        self.reg_cmds
            .set(81, npu_op(OP_REG_DPU, 0x0, DPU_OUT_CVT_SHIFT));
        self.reg_cmds
            .set(82, npu_op(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_0));
        self.reg_cmds
            .set(83, npu_op(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_1));
        self.reg_cmds
            .set(84, npu_op(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_2));
        self.reg_cmds
            .set(85, npu_op(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_3));
        self.reg_cmds
            .set(86, npu_op(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_4));
        self.reg_cmds
            .set(87, npu_op(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_5));
        self.reg_cmds
            .set(88, npu_op(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_6));
        self.reg_cmds
            .set(89, npu_op(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_7));
        value = (dpu.surf_add & 0xFFFFFFF) << 4;
        self.reg_cmds
            .set(90, npu_op(OP_REG_DPU, value, DPU_SURFACE_ADD));
        self.reg_cmds.set(91, npu_op(OP_REG_DPU, 0x0, DPU_40C4));
        self.reg_cmds
            .set(92, npu_op(OP_REG_DPU, 0x0, DPU_LUT_ACCESS_CFG));
        self.reg_cmds
            .set(93, npu_op(OP_REG_DPU, 0x0, DPU_LUT_ACCESS_DATA));
        self.reg_cmds.set(94, npu_op(OP_REG_DPU, 0x0, DPU_LUT_CFG));
        self.reg_cmds.set(95, npu_op(OP_REG_DPU, 0x0, DPU_LUT_INFO));
        self.reg_cmds
            .set(96, npu_op(OP_REG_DPU, 0x0, DPU_LUT_LE_START));
        self.reg_cmds
            .set(97, npu_op(OP_REG_DPU, 0x0, DPU_LUT_LE_END));
        self.reg_cmds
            .set(98, npu_op(OP_REG_DPU, 0x0, DPU_LUT_LO_START));
        self.reg_cmds
            .set(99, npu_op(OP_REG_DPU, 0x0, DPU_LUT_LO_END));
        self.reg_cmds
            .set(100, npu_op(OP_REG_DPU, 0x0, DPU_LUT_LE_SLOPE_SCALE));
        self.reg_cmds
            .set(101, npu_op(OP_REG_DPU, 0x0, DPU_LUT_LE_SLOPE_SHIFT));
        self.reg_cmds
            .set(102, npu_op(OP_REG_DPU, 0x0, DPU_LUT_LO_SLOPE_SCALE));
        self.reg_cmds
            .set(103, npu_op(OP_REG_DPU, 0x0, DPU_LUT_LO_SLOPE_SHIFT));
        self.reg_cmds.set(104, npu_op(OP_NONE, 0x0, 0x0));
        self.reg_cmds
            .set(105, npu_op(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS));
        self.reg_cmds.set(106, npu_op(OP_40, 0x0, 0x0));
        self.reg_cmds.set(
            107,
            npu_op(
                OP_ENABLE,
                PC_ENABLE_DPU | PC_ENABLE_CNA | PC_ENABLE,
                PC_OPERATION_ENABLE,
            ),
        );
    }

    pub fn set_a_b(&mut self, a: &[i8], b: &[i8]) {
        assert_eq!(a.len(), self.m as usize * self.k as usize);
        assert_eq!(b.len(), self.k as usize * self.n as usize);
        let k = self.k as i32;
        let n = self.n as i32;
        for nn in 1..=n {
            for kk in 1..=k {
                let idx = weight_int8(k, nn, kk) as usize;
                let src = ((nn - 1) * k + (kk - 1)) as usize;
                self.weight.set(idx, b[src]);
            }
        }

        let m = self.m as i32;
        let k = self.k as i32;
        for mm in 1..=m {
            for kk in 1..=k {
                let idx = feature_data(k, m, 1, 16, kk, mm, 1) as usize;
                let src = ((mm - 1) * k + (kk - 1)) as usize;
                self.input.set(idx, a[src]);
            }
        }
    }
}

fn weight_int8(C: i32, k: i32, c: i32) -> i32 {
    let kpg = (k - 1) / 32;
    let cpg = (c - 1) / 32;
    let mut dst = (cpg * 32) * 32 + (kpg * 32 * C);
    dst += (c - 1) % 32 + ((k - 1) % 32) * 32;
    dst
}

pub fn feature_data(C: i32, H: i32, W: i32, C2: i32, c: i32, h: i32, w: i32) -> i32 {
    let plane = (c - 1) / C2;
    let src = plane * H * W * C2;
    let offset = (c - 1) % C2;
    let pos = src + C2 * ((h - 1) * W + (w - 1)) + offset;
    pos
}

pub struct RknpuSubmitK {
    pub flags: JobMode,
    pub tasks: DVec<RknpuTask>,
    pub task_base_addr: u32,
    pub ops: Vec<Matmul>,
}

impl RknpuSubmitK {
    pub fn new(mut ops: Vec<Matmul>) -> Self {
        let mut tasks =
            DVec::zeros(u32::MAX as _, ops.len(), 0x1000, Direction::Bidirectional).unwrap();

        for (i, op) in ops.iter_mut().enumerate() {
            // Prepare task for each matmul operation
            tasks.set(i, op.as_task());
        }

        Self {
            flags: JobMode::PC | JobMode::BLOCK | JobMode::PINGPONG,
            tasks,
            ops,
            task_base_addr: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubmitBase {
    pub flags: JobMode,
    pub task_base_addr: u32,
    pub core_idx: usize,
    pub int_mask: u32,
    pub int_clear: u32,
    pub regcfg_amount: u32,
}

#[derive(Debug, Clone)]
pub struct SubmitRef {
    pub base: SubmitBase,
    pub task_number: usize,
    pub regcmd_base_addr: u32,
}

pub struct Submit {
    pub base: SubmitBase,
    pub regcmd_all: DVec<u64>,
    pub tasks: Vec<Operation>,
}

impl Submit {
    pub fn new(tasks: Vec<Operation>) -> Self {
        let base = SubmitBase {
            flags: JobMode::PC | JobMode::BLOCK | JobMode::PINGPONG,
            task_base_addr: 0,
            core_idx: 0,
            int_mask: 0x300, // wait for DPU to finish
            int_clear: 0x1ffff,
            regcfg_amount: tasks[0].reg_amount(),
        };

        let regcmd_all: DVec<u64> = DVec::zeros(
            u32::MAX as _,
            base.regcfg_amount as usize * tasks.len(),
            0x1000,
            Direction::Bidirectional,
        )
        .unwrap();

        assert!(
            regcmd_all.bus_addr() <= u32::MAX as u64,
            "regcmd base address exceeds u32"
        );

        let amount = base.regcfg_amount as usize;
        for (i, task) in tasks.iter().enumerate() {
            let regcmd = unsafe {
                core::slice::from_raw_parts_mut(regcmd_all.as_ptr().add(i * amount), amount)
            };
            task.fill_regcmd(regcmd);
        }
        regcmd_all.confirm_write_all();

        Self {
            base,
            regcmd_all,
            tasks,
        }
    }

    pub fn as_ref(&self) -> SubmitRef {
        SubmitRef {
            base: self.base.clone(),
            task_number: self.tasks.len(),
            regcmd_base_addr: self.regcmd_all.bus_addr() as _,
        }
    }
}
