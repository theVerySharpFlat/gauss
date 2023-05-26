#[derive(Debug, Copy, Clone)]
pub enum InitError {
    NoDevices,
    NoVulkanDevices,
    NoComputeQueue,
    LogicalDeviceCreationFailure,
    QueueCreationFailure,
    LibraryNotFound,
    InstanceCreateFailed,
    DebugMessengerCreationFailed,
    PhysicalDeviceQueryFailed,
}
