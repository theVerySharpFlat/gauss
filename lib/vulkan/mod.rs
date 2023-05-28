use self::{
    device::{initialize_device, DeviceInfo},
    init_error::InitError,
    instance::{create_instance, InstanceInfo},
};

mod device;
mod init_error;
mod instance;

pub struct ComputeManager {
    instance_info: InstanceInfo,
    device_info: DeviceInfo,
}

impl Drop for ComputeManager {
    fn drop(&mut self) {
        unsafe {
            self.device_info.device.device_wait_idle().unwrap();
            self.device_info.device.destroy_device(None);
            if self.instance_info.debug_utils_loader.is_some() {
                self.instance_info
                    .debug_utils_loader
                    .as_ref()
                    .unwrap()
                    .destroy_debug_utils_messenger(
                        self.instance_info.debug_messenger.unwrap(),
                        None,
                    );
            }
            self.instance_info.instance.destroy_instance(None);
        }
    }
}

pub fn compute_init() -> Result<ComputeManager, InitError> {
    let instance_info = create_instance(true)?;
    let device_info = initialize_device(&instance_info, true)?;

    Ok(ComputeManager {
        instance_info,
        device_info,
    })
}
