use std::mem::MaybeUninit;

use self::{
    device::{initialize_device, DeviceInfo},
    init_error::InitError,
    instance::{create_instance, InstanceInfo},
};

pub use allocation_strategy::Tensor;
use gpu_allocator::vulkan::Allocator;
pub use gpu_task::WorkGroupSize;

mod allocation_strategy;
mod command_buffer_util;
mod device;
mod gpu_task;
mod init_error;
mod instance;
mod pipeline;

pub struct ComputeManager {
    instance_info: InstanceInfo,
    device_info: DeviceInfo,
    allocator: allocation_strategy::Allocator,
    current_tensor_id: u32,
}

impl Drop for ComputeManager {
    fn drop(&mut self) {
        unsafe {
            self.device_info.device.device_wait_idle().unwrap();

            self.device_info
                .device
                .destroy_command_pool(self.device_info.compute_pool, None);

            // Free the VkMemory allocations made by the allocator
            #[allow(invalid_value)]
            let mut dummy_allocator: Allocator = MaybeUninit::zeroed().assume_init();

            std::mem::swap(&mut self.allocator.vulkan_allocator, &mut dummy_allocator);
            drop(dummy_allocator);

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
    let allocator = match allocation_strategy::Allocator::new(&instance_info, &device_info) {
        Ok(a) => a,
        Err(e) => {
            println!("Failed to create allocator! Error: {:?}", e);
            return Err(InitError::AllocatorCreationFailure);
        }
    };

    Ok(ComputeManager {
        instance_info,
        device_info,
        allocator,
        current_tensor_id: 0,
    })
}
