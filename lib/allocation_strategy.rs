use std::ptr;

use ash::vk;
use ash::vk::{BufferCreateFlags, BufferCreateInfo, BufferUsageFlags, SharingMode, StructureType};

use gpu_allocator::vulkan::{Allocation, AllocationScheme};
use gpu_allocator::MemoryLocation;
use gpu_allocator::{
    vulkan::{AllocationCreateDesc, Allocator as VulkanAllocator, AllocatorCreateDesc},
    AllocatorDebugSettings,
};

use ndarray::prelude::*;

use crate::AllocatorLogConfig;

use super::ComputeManager;
use super::{device::DeviceInfo, instance::InstanceInfo};

pub struct Allocator {
    pub(super) vulkan_allocator: VulkanAllocator,
}

pub struct Buffer {
    pub(super) buffer: vk::Buffer,
    pub(super) allocation: Allocation,
}

pub struct Tensor {
    pub(super) id: u32,
    pub(super) readback_enabled: bool,

    local_data: Array<f32, Ix1>,
}

#[derive(Debug, Clone, Copy)]
pub enum AllocationError {
    AllocatorCreationFailure,
    BufferCreationFailure,
    MemoryAllocationError,
    MemoryBindFailure,
}

impl ComputeManager {
    pub fn create_tensor(&self, data: Array<f32, Ix1>, enable_readback: bool) -> Tensor {
        Tensor {
            id: self.current_tensor_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            readback_enabled: enable_readback,
            local_data: data,
        }
    }
}

impl Tensor {
    pub fn data(&self) -> &Array<f32, Ix1> {
        &self.local_data
    }

    pub fn data_mut(&mut self) -> &mut Array<f32, Ix1> {
        &mut self.local_data
    }
}

impl Allocator {
    pub fn new(
        instance_info: &InstanceInfo,
        device_info: &DeviceInfo,
        log_config: Option<AllocatorLogConfig>,
    ) -> Result<Self, AllocationError> {
        let vulkan_allocator = match VulkanAllocator::new(&AllocatorCreateDesc {
            instance: instance_info.instance.clone(),
            device: device_info.device.clone(),
            physical_device: device_info.physical_device,
            debug_settings: if let Some(cfg) = log_config {
                AllocatorDebugSettings {
                    log_memory_information: cfg.log_memory_information,
                    log_leaks_on_shutdown: cfg.log_leaks_on_shutdown,
                    store_stack_traces: cfg.store_stack_traces,
                    log_allocations: cfg.log_allocations,
                    log_frees: cfg.log_frees,
                    log_stack_traces: cfg.log_stack_traces,
                }
            } else {
                AllocatorDebugSettings::default()
            },
            buffer_device_address: false,
        }) {
            Ok(a) => a,
            Err(e) => {
                log::error!("Failed to create allocator! Error: \"{}\"", e);
                return Err(AllocationError::AllocatorCreationFailure);
            }
        };

        Ok(Allocator { vulkan_allocator })
    }

    pub fn allocate_buffer(
        &mut self,
        device_info: &DeviceInfo,
        size: u64,
        usage: BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
        queue_family: u32,
    ) -> Result<Buffer, AllocationError> {
        let queue_families = [queue_family];

        let buffer_create_info = BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: BufferCreateFlags::empty(),
            size,
            usage,
            sharing_mode: SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: queue_families.as_ptr(),
        };

        let buffer = unsafe {
            match device_info.device.create_buffer(&buffer_create_info, None) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("Failed to allocate buffer with error {}", e);
                    return Err(AllocationError::BufferCreationFailure);
                }
            }
        };

        let buffer_memory_requirements = unsafe {
            device_info
                .device
                .get_buffer_memory_requirements(buffer.clone())
        };

        let buffer_allocation = match self.vulkan_allocator.allocate(&AllocationCreateDesc {
            name,
            requirements: buffer_memory_requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => {
                log::error!("Failed to allocate backing memory for buffer! Error: {}", e);
                return Err(AllocationError::MemoryAllocationError);
            }
        };

        unsafe {
            match device_info.device.bind_buffer_memory(
                buffer,
                buffer_allocation.memory(),
                buffer_allocation.offset(),
            ) {
                Ok(_) => (),
                Err(e) => {
                    log::error!("Failed to bind buffer memory! Error: {}", e);
                    return Err(AllocationError::MemoryBindFailure);
                }
            };
        }

        Ok(Buffer {
            buffer,
            allocation: buffer_allocation,
        })
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        // evil
        #[allow(invalid_value)]
        let mut swapped_out: VulkanAllocator = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
        std::mem::swap(&mut swapped_out, &mut self.vulkan_allocator);

        drop(swapped_out); 
    }
}
