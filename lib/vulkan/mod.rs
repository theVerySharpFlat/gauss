use self::{
    init_error::InitError,
    instance::{create_instance, InstanceInfo}, device::initialize_device,
};

mod device;
mod init_error;
mod instance;

pub struct ComputeManager {
    instance_info: InstanceInfo,
}

pub fn compute_init() -> Result<ComputeManager, InitError> {
    let instance_info = create_instance(true)?;
    initialize_device(&instance_info, true);

    Ok(ComputeManager { instance_info })
}
