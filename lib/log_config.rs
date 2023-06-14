#[derive(Debug, Copy, Clone)]
pub struct ValidationLayerLogConfig {
    pub log_errors: bool,
    pub log_warnings: bool,
    pub log_verbose_info: bool,
}

#[derive(Debug, Copy, Clone)]
pub struct AllocatorLogConfig {
    pub log_memory_information: bool,
    pub log_leaks_on_shutdown: bool,
    pub store_stack_traces: bool,
    pub log_allocations: bool,
    pub log_frees: bool,
    pub log_stack_traces: bool,
}

#[derive(Debug, Copy, Clone)]
pub struct LogConfig {
    pub validation_config: Option<ValidationLayerLogConfig>,
    pub allocator_config: Option<AllocatorLogConfig>,
}
