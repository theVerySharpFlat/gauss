use std::{
    borrow::Cow,
    ffi::{c_void, CStr, CString, c_char},
    ptr,
};

use ash::{
    extensions::ext::DebugUtils,
    vk::{
        self, ApplicationInfo, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT, InstanceCreateFlags,
        InstanceCreateInfo, KhrPortabilityEnumerationFn, KhrGetPhysicalDeviceProperties2Fn, StructureType,
    },
    Entry, Instance,
};

use super::init_error::InitError;

// #[derive(Debug)]
pub struct InstanceInfo {
    pub instance: Instance,
    pub debug_messenger: Option<DebugUtilsMessengerEXT>,
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
        "{message_severity:?}: {message_type:?} [{message_id_name} ({message_id_number})] : {message}",
    );

    vk::FALSE
}

fn get_debug_utils_messenger_info() -> DebugUtilsMessengerCreateInfoEXT {
    let message_severity = DebugUtilsMessageSeverityFlagsEXT::default()
        //| DebugUtilsMessageSeverityFlagsEXT::INFO
        | DebugUtilsMessageSeverityFlagsEXT::WARNING
        | DebugUtilsMessageSeverityFlagsEXT::ERROR
        | DebugUtilsMessageSeverityFlagsEXT::VERBOSE;

    let message_type = DebugUtilsMessageTypeFlagsEXT::default()
        | DebugUtilsMessageTypeFlagsEXT::GENERAL
        | DebugUtilsMessageTypeFlagsEXT::VALIDATION
        | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE;

    DebugUtilsMessengerCreateInfoEXT::builder()
        .pfn_user_callback(Some(vulkan_debug_callback))
        .message_severity(message_severity)
        .message_type(message_type)
        .build()
}

pub fn create_instance(enable_validation: bool) -> Result<InstanceInfo, InitError> {
    unsafe {
        let entry = Entry::linked();

        let app_name = CString::new("ICompute_APP").unwrap();
        let engine_name = CString::new("ICompute_ENGINE").unwrap();
        let app_info = ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(vk::make_api_version(1, 0, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(1, 0, 0, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0))
            .build();

        let mut extension_names = Vec::new();
        #[cfg(any(target_os = "macos"))]
        {
            extension_names.push(KhrPortabilityEnumerationFn::name());
            extension_names.push(KhrGetPhysicalDeviceProperties2Fn::name());
        }

        if enable_validation {
            extension_names.push(DebugUtils::name());
        }

        let layer_names = [
            CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0")
        ];

        let mut instance_flags = InstanceCreateFlags::default();
        #[cfg(any(target_os = "macos"))]
        {
            instance_flags |= InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let layer_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|item| item.as_ptr())
            .collect();

        let extension_names_raw: Vec<*const i8> = extension_names
            .iter()
            .map(|item| (*item).as_ptr())
            .collect();

        let debug_messenger_info = get_debug_utils_messenger_info();

        let instance_create_info = InstanceCreateInfo {
            s_type: StructureType::INSTANCE_CREATE_INFO,
            p_next: if enable_validation {
                &debug_messenger_info as *const DebugUtilsMessengerCreateInfoEXT as *const c_void
            } else {
                ptr::null()
            },
            flags: instance_flags,
            p_application_info: &app_info,
            enabled_layer_count: layer_names.len() as u32,
            pp_enabled_layer_names: layer_names_raw.as_ptr(),
            enabled_extension_count: extension_names.len() as u32,
            pp_enabled_extension_names: extension_names_raw.as_ptr(),
        };

        let instance = match entry.create_instance(&instance_create_info, None) {
            Ok(instance) => instance,
            Err(e) => {
                println!("Instance creation failed with error \"{}\"", e);
                return Err(InitError::InstanceCreateFailed);
            }
        };

        let mut debug_messenger: Option<DebugUtilsMessengerEXT> = None;
        if enable_validation {
            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            debug_messenger = match debug_utils_loader
                .create_debug_utils_messenger(&debug_messenger_info, None)
            {
                Ok(messenger) => Some(messenger),
                Err(e) => {
                    println!(
                        "Failed to create debug messenger! Creation failed with error \"{}\"",
                        e
                    );
                    return Err(InitError::DebugMessengerCreationFailed);
                }
            };
        }

        Ok(InstanceInfo {
            debug_messenger,
            instance,
        })
    }
}
