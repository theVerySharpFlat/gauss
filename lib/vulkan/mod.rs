use self::{
    init_error::InitError,
    instance::{create_instance, InstanceInfo},
};

mod device;
mod init_error;
mod instance;

pub struct ComputeManager {
    instance_info: InstanceInfo,
}

pub fn compute_init() -> Result<ComputeManager, InitError> {
    let instance_info = create_instance(true)?;

    Ok(ComputeManager { instance_info })
}
