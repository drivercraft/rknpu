#[derive(thiserror::Error, Debug)]
pub enum RknpuError {
    InvalidArgument,
    Timeout,
    NoMemory,
    Unsupported,
}
