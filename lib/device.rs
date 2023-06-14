use std::{
    cmp::Ordering,
    ffi::{CStr, CString},
    ptr,
};

use ash::{
    vk::{
        self, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo, DeviceCreateFlags,
        DeviceCreateInfo, DeviceQueueCreateFlags, DeviceQueueCreateInfo, PhysicalDevice,
        PhysicalDeviceFeatures, PhysicalDeviceType, Queue, QueueFamilyProperties, QueueFlags,
        StructureType,
    },
    Device, Instance,
};

use super::{init_error::InitError, instance::InstanceInfo};

#[derive(Clone)]
pub struct DeviceInfo {
    pub device: Device,
    pub compute_queue: Queue,
    pub physical_device: PhysicalDevice,
    pub queue_indices: QueueFamilyInfo,

    pub compute_pool: CommandPool,
}

fn score_device(instance: &Instance, physical_device: PhysicalDevice) -> Option<u32> {
    let mut score = 0;

    unsafe {
        let device_properties = instance.get_physical_device_properties(physical_device);

        score += match device_properties.device_type {
            PhysicalDeviceType::DISCRETE_GPU => 10,
            PhysicalDeviceType::INTEGRATED_GPU => 5,
            _ => 0,
        };

        let compute_queue_count: u32 = instance
            .get_physical_device_queue_family_properties(physical_device)
            .iter()
            .filter(|queue_info| queue_info.queue_count > 0)
            .map(|val| -> u32 {
                if val.queue_flags.contains(QueueFlags::COMPUTE) {
                    1
                } else {
                    0
                }
            })
            .sum();

        if compute_queue_count == 0 {
            return None;
        }
        score += compute_queue_count * 5;
    }

    Some(score)
}

#[derive(Clone)]
pub struct QueueFamilyInfo {
    pub compute_queue: Option<u32>,
}

impl QueueFamilyInfo {
    fn complete(self: &Self) -> bool {
        return self.compute_queue.is_some();
    }
}

fn load_queue_family_info(instance: &Instance, physical_device: PhysicalDevice) -> QueueFamilyInfo {
    unsafe {
        let score_queue = |info: &QueueFamilyProperties| {
            if info.queue_flags.contains(QueueFlags::COMPUTE) {
                if info.queue_flags.contains(QueueFlags::GRAPHICS) {
                    1
                } else {
                    2
                }
            } else {
                0
            }
        };

        let queue_family_infos =
            instance.get_physical_device_queue_family_properties(physical_device.clone());

        let best_queue = queue_family_infos
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let b_score = score_queue(b);
                score_queue(a).cmp(&b_score)
            });

        let compute_queue = match best_queue {
            Some((queue, _)) => Some(queue as u32),
            None => None,
        };

        QueueFamilyInfo { compute_queue }
    }
}

fn create_compute_pool(device: &Device, queue_index: u32) -> Result<CommandPool, InitError> {
    let create_info = CommandPoolCreateInfo {
        s_type: StructureType::COMMAND_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        queue_family_index: queue_index,
    };

    unsafe {
        Ok(match device.create_command_pool(&create_info, None) {
            Ok(p) => p,
            Err(e) => {
                log::error!("Failed to create command pool! Error: {}", e);
                return Err(InitError::ComputePoolCreationFailure);
            }
        })
    }
}

pub fn log_device_info(instance: &Instance, device: &Device, physical_device: PhysicalDevice) {
    unsafe {
        let mut physical_device_properties =
            instance.get_physical_device_properties(physical_device);
        let api_version = physical_device_properties.api_version;

        log::info!("Device creation succeeded with: ");
        log::info!(
            "\tGPU_NAME: \"{}\"",
            CStr::from_ptr(physical_device_properties.device_name.as_mut_ptr())
                .to_str()
                .unwrap_or("DEVICE_NAME_RETRIEVE_ERROR")
        );
        log::info!(
            "\tGPU_TYPE: \"{:?}\"",
            physical_device_properties.device_type
        );
        log::info!(
            "\tAPI_VERSION: {}.{}.{}",
            vk::api_version_major(api_version),
            vk::api_version_minor(api_version),
            vk::api_version_patch(api_version)
        );
    }
}

pub fn initialize_device(
    instance_info: &InstanceInfo,
    enable_validation: bool,
) -> Result<DeviceInfo, InitError> {
    unsafe {
        let physical_devices = match instance_info.instance.enumerate_physical_devices() {
            Ok(devices) => devices,
            Err(err) => {
                log::error!(
                    "Failed to query for physical devices due to error \"{}\"",
                    err
                );
                return Err(InitError::PhysicalDeviceQueryFailed);
            }
        };

        let optimal_device_opt = physical_devices.iter().max_by(|a, b| {
            let b_score = score_device(&instance_info.instance, **b);
            let a_score = score_device(&instance_info.instance, **a);

            if b_score == a_score && a_score == None {
                Ordering::Equal
            } else if b_score == None {
                Ordering::Greater
            } else if a_score == None {
                Ordering::Less
            } else {
                a_score.cmp(&b_score)
            }
        });

        if optimal_device_opt == None {
            log::error!("Failed to find adequate device!");
            return Err(InitError::NoDevices);
        }

        let physical_device = optimal_device_opt.unwrap();

        let queue_family_info =
            load_queue_family_info(&instance_info.instance, physical_device.clone());
        if !queue_family_info.complete() {
            return Err(InitError::NoComputeQueue);
        }

        let queue_prior = [1.0 as f32];

        let mut queue_create_infos = Vec::new();
        queue_create_infos.push(DeviceQueueCreateInfo {
            s_type: StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: DeviceQueueCreateFlags::empty(),
            queue_family_index: queue_family_info.compute_queue.unwrap(),
            queue_count: 1,
            p_queue_priorities: queue_prior.as_ptr(),
        });

        let physical_device_features = PhysicalDeviceFeatures {
            ..Default::default()
        };

        #[allow(unused_mut)]
        let mut device_extensions: Vec<*const i8> = vec![];
        #[cfg(any(target_os = "macos"))]
        {
            device_extensions
                .push(CStr::from_bytes_with_nul_unchecked(b"VK_KHR_portability_subset\0").as_ptr());
        }

        let layer_names =
            [CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()];

        let device_create_info = DeviceCreateInfo {
            s_type: StructureType::DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            flags: DeviceCreateFlags::default(),
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_layer_count: 1,
            pp_enabled_layer_names: if enable_validation {
                layer_names.as_ptr()
            } else {
                ptr::null()
            },
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            p_enabled_features: &physical_device_features,
        };

        let device = match instance_info.instance.create_device(
            physical_device.clone(),
            &device_create_info,
            None,
        ) {
            Ok(dev) => dev,
            Err(e) => {
                log::error!("Device creation failed with error \"{}\"", e);
                return Err(InitError::LogicalDeviceCreationFailure);
            }
        };

        log_device_info(&instance_info.instance, &device, *physical_device);

        let compute_queue = device.get_device_queue(queue_family_info.compute_queue.unwrap(), 0);

        return Ok(DeviceInfo {
            device: device.clone(),
            compute_queue,
            physical_device: *physical_device,
            queue_indices: load_queue_family_info(&instance_info.instance, physical_device.clone()),
            compute_pool: create_compute_pool(&device, queue_family_info.compute_queue.unwrap())?,
        });
    }
}
