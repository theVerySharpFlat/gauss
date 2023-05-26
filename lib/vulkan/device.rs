use std::{cmp::Ordering, ptr, ffi::CStr};

use ash::{
    vk::{PhysicalDevice, PhysicalDeviceType, QueueFlags, QueueFamilyProperties, DeviceCreateInfo, StructureType, DeviceCreateFlags, DeviceQueueCreateInfo, DeviceQueueCreateFlags, PhysicalDeviceFeatures, Handle},
    Instance,
};

use super::{init_error::InitError, instance::InstanceInfo};

#[derive(Debug)]
pub struct DeviceInfo {}

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

struct QueueFamilyInfo {
    compute_queue: Option<u32>,
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

        println!("queue family count: {}", queue_family_infos.len());

        let best_queue = queue_family_infos
            .iter()
            .enumerate() 
            .max_by(|(_, a), (_, b)| {
                let b_score = score_queue(b);
                score_queue(a).cmp(&b_score)
            });

        let compute_queue = match best_queue {
            Some((queue, _)) => {
                Some(queue as u32)
            },
            None => None
        };

        QueueFamilyInfo { compute_queue }
    }
}

pub fn initialize_device(instance_info: &InstanceInfo) -> Result<DeviceInfo, InitError> {
    unsafe {
        let physical_devices = match instance_info.instance.enumerate_physical_devices() {
            Ok(devices) => devices,
            Err(err) => {
                println!(
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
            println!("Failed to find adequate device!");
            return Err(InitError::NoDevices);
        }

        let physical_device = optimal_device_opt.unwrap();

        let queue_family_info = load_queue_family_info(&instance_info.instance, physical_device.clone());
        if !queue_family_info.complete() {
            return Err(InitError::NoComputeQueue);
        }

        let queue_prior = [1.0 as f32];

        println!("queue index: {}", queue_family_info.compute_queue.unwrap());

        let mut queue_create_infos = Vec::new();
        queue_create_infos.push(DeviceQueueCreateInfo {
            s_type: StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: DeviceQueueCreateFlags::empty(),
            queue_family_index: 0,//queue_family_info.compute_queue.unwrap(),
            queue_count: 1,
            p_queue_priorities: queue_prior.as_ptr(),
        });

        let physical_device_features = Vec::<PhysicalDeviceFeatures>::new();

        let device_extensions = [
            CStr::from_bytes_with_nul_unchecked(b"VK_KHR_portability_subset\0").as_ptr()
       ];

        let layer_names = [
            CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()
        ];

        let device_create_info = DeviceCreateInfo {
            s_type: StructureType::DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            flags: DeviceCreateFlags::default(),
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_layer_count: 1,
            pp_enabled_layer_names: layer_names.as_ptr(),
            enabled_extension_count: 1,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            p_enabled_features: physical_device_features.as_ptr(),
        };

        let device = instance_info.instance.create_device(physical_device.clone(), &device_create_info, None);
        println!("right here");
    }

    Err(InitError::NoDevices)
}
