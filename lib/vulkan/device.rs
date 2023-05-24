use super::init_error::InitError;

#[derive(Debug)]
pub struct DeviceInfo {
}

fn select_physical_device(instance: Arc<Instance>) -> Result<Arc<PhysicalDevice>, InitError> {
}

pub fn load_device() -> Result<DeviceInfo, InitError> {
}
