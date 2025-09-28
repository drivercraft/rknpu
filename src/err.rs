//! Error types for RKNPU operations

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RknpuError {
    /// Invalid parameter or argument
    InvalidParameter,
    /// Operation timed out
    Timeout,
    /// Out of memory
    OutOfMemory,
    /// Operation not supported
    NotSupported,
    /// Device is busy
    DeviceBusy,
    /// Device is not ready or not initialized
    DeviceNotReady,
    /// General device error
    DeviceError,
    /// Hardware fault
    HardwareFault,
    /// IOMMU error
    IommuError,
    /// DMA error
    DmaError,
    /// Task submission error
    TaskError,
    /// Memory management error
    MemoryError,
    /// Invalid handle
    InvalidHandle,
    /// Resource temporarily unavailable
    TryAgain,
    /// Operation interrupted
    Interrupted,
    /// Permission denied
    PermissionDenied,
    /// Internal error
    InternalError,
}
