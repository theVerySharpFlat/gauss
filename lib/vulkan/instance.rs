use std::{ffi::{CString, CStr}, ptr, borrow::Cow};

use ash::vk::{self, InstanceCreateInfo, StructureType, InstanceCreateFlags};

use super::init_error::InitError;

#[derive(Debug)]
pub struct InstanceInfo {
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );

    vk::FALSE
}

pub fn create_instance() -> Result<InstanceInfo, InitError> {
    let app_name = CString::new("ICompute_APP").unwrap();
    let engine_name = CString::new("ICompute_ENGINE").unwrap();
    let app_info = ApplicationInfo::builder()
        .application_name(&app_name)
        .application_version(vk::make_api_version(1, 0, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(1, 0, 0, 0))
        .api_version(vk::make_api_version(1, 0, 0, 0))
        .build();

    let debug_messenger_info;

    let instance_info = InstanceCreateInfo {
        s_type: StructureType::INSTANCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
        p_application_info: &app_info,
        enabled_layer_count: 1,
        pp_enabled_layer_names: todo!(),
        enabled_extension_count: todo!(),
        pp_enabled_extension_names: todo!(),
    }
}
