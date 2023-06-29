use std::{
    mem::MaybeUninit,
    sync::{atomic::AtomicU32, Arc, RwLock},
};

use self::{
    device::{initialize_device, DeviceInfo},
    init_error::InitError,
    instance::{create_instance, InstanceInfo},
};

use allocation_strategy::Allocator;
pub use allocation_strategy::Tensor;
pub use gpu_task::WorkGroupSize;
pub use log_config::AllocatorLogConfig;
pub use log_config::LogConfig;
pub use log_config::ValidationLayerLogConfig;

mod allocation_strategy;
mod command_buffer_util;
mod device;
mod gpu_task;
mod init_error;
mod instance;
mod log_config;
mod pipeline;

pub struct ComputeManager {
    instance_info: InstanceInfo,
    device_info: DeviceInfo,
    allocator: Arc<RwLock<allocation_strategy::Allocator>>,
    current_tensor_id: AtomicU32,
}

impl Drop for ComputeManager {
    fn drop(&mut self) {
        unsafe {
            self.device_info.device.device_wait_idle().unwrap();

            self.device_info
                .device
                .destroy_command_pool(self.device_info.compute_pool, None);

            // Free the VkMemory allocations made by the allocator
            if let Ok(mut allocator) = self.allocator.write() {
                #[allow(invalid_value)]
                let mut to_drop: Allocator = MaybeUninit::zeroed().assume_init();
                std::mem::swap(&mut (*allocator), &mut to_drop);

                drop(to_drop);
            }

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

pub fn compute_init(log_config: LogConfig) -> Result<Arc<ComputeManager>, InitError> {
    env_logger::init();

    log::trace!("Hello world");

    let instance_info = create_instance(log_config.validation_config)?;
    let device_info = initialize_device(&instance_info, true)?;
    let allocator = match allocation_strategy::Allocator::new(
        &instance_info,
        &device_info,
        log_config.allocator_config,
    ) {
        Ok(a) => a,
        Err(e) => {
            log::error!("Failed to create allocator! Error: {:?}", e);
            return Err(InitError::AllocatorCreationFailure);
        }
    };

    Ok(Arc::new(ComputeManager {
        instance_info,
        device_info,
        allocator: Arc::new(RwLock::new(allocator)),
        current_tensor_id: AtomicU32::new(0),
    }))
}
