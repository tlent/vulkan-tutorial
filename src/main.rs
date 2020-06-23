use std::ffi::{c_void, CStr, CString};

use ash::extensions::ext::DebugUtils;
use ash::version::{EntryV1_0, InstanceV1_0};
use ash::{vk, Entry, Instance};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

#[cfg(debug_assertions)]
const VALIDATION_ENABLED: bool = true;
#[cfg(not(debug_assertions))]
const VALIDATION_ENABLED: bool = false;

fn main() {
    let (window, event_loop) = init_window();
    let app = HelloTriangleApp::new(&window);
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => app.run(),
            _ => (),
        }
    });
}

fn init_window() -> (Window, EventLoop<()>) {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .with_title("Vulkan")
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();
    (window, event_loop)
}

struct HelloTriangleApp {
    entry: Entry,
    instance: Instance,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    physical_device: vk::PhysicalDevice,
}

impl HelloTriangleApp {
    pub fn new(window: &Window) -> Self {
        let entry = Entry::new().unwrap();
        let instance = Self::create_instance(&entry, window);
        let debug_messenger = if VALIDATION_ENABLED {
            Some(Self::setup_debug_messenger(&entry, &instance))
        } else {
            None
        };
        let physical_device = Self::select_physical_device(&instance);
        Self {
            entry,
            instance,
            debug_messenger,
            physical_device,
        }
    }

    fn create_instance(entry: &Entry, window: &Window) -> Instance {
        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_application_name: CString::new("Hello Triangle").unwrap().as_ptr(),
            application_version: vk::make_version(1, 0, 0),
            p_engine_name: CString::new("No Engine").unwrap().as_ptr(),
            engine_version: vk::make_version(1, 0, 0),
            api_version: vk::make_version(1, 0, 0),
            ..Default::default()
        };
        let mut required_extensions = ash_window::enumerate_required_extensions(window).unwrap();
        if VALIDATION_ENABLED {
            required_extensions.push(DebugUtils::name());
        }
        if let Err(extension_name) = Self::check_extension_support(entry, &required_extensions) {
            panic!(
                "Required extension {} is not supported.",
                extension_name.to_string_lossy()
            );
        }
        let extension_refs: Vec<_> = required_extensions.iter().map(|e| e.as_ptr()).collect();
        let mut create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: extension_refs.len() as u32,
            pp_enabled_extension_names: extension_refs.as_ptr(),
            ..Default::default()
        };
        let validation_layers =
            vec![CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap()];
        let layer_refs: Vec<_> = validation_layers.iter().map(|l| l.as_ptr()).collect();
        let debug_messenger_create_info = debug_messenger_create_info();
        if VALIDATION_ENABLED {
            if let Err(layer_name) = Self::check_validation_layer_support(entry, &validation_layers)
            {
                panic!(
                    "Validation layer {} is not supported.",
                    layer_name.to_string_lossy()
                );
            }
            create_info.enabled_layer_count = layer_refs.len() as u32;
            create_info.pp_enabled_layer_names = layer_refs.as_ptr();
            create_info.p_next = &debug_messenger_create_info
                as *const vk::DebugUtilsMessengerCreateInfoEXT
                as *const c_void;
        }
        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };
        instance
    }

    fn check_extension_support<'a>(
        entry: &Entry,
        required_extensions: &[&'a CStr],
    ) -> Result<(), &'a CStr> {
        let supported_extensions = entry.enumerate_instance_extension_properties().unwrap();
        let extension_refs: Vec<_> = supported_extensions
            .iter()
            .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
            .collect();
        for e in required_extensions.iter() {
            if !extension_refs.contains(e) {
                return Err(e);
            }
        }
        Ok(())
    }

    fn check_validation_layer_support<'a>(
        entry: &Entry,
        required_layers: &[&'a CStr],
    ) -> Result<(), &'a CStr> {
        let supported_layers = entry.enumerate_instance_layer_properties().unwrap();
        let layer_refs: Vec<_> = supported_layers
            .iter()
            .map(|l| unsafe { CStr::from_ptr(l.layer_name.as_ptr()) })
            .collect();
        for l in required_layers.iter() {
            if !layer_refs.contains(l) {
                return Err(l);
            }
        }
        Ok(())
    }

    fn setup_debug_messenger(entry: &Entry, instance: &Instance) -> vk::DebugUtilsMessengerEXT {
        let create_info = debug_messenger_create_info();
        let debug_utils = DebugUtils::new(entry, instance);
        unsafe {
            debug_utils
                .create_debug_utils_messenger(&create_info, None)
                .unwrap()
        }
    }

    fn select_physical_device(instance: &Instance) -> vk::PhysicalDevice {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        devices
            .into_iter()
            .find(|&d| Self::is_device_suitable(instance, d))
            .expect("Failed to find a suitable GPU")
    }

    fn is_device_suitable(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        Self::find_queue_familes(instance, device).is_complete()
    }

    fn find_queue_familes(instance: &Instance, device: vk::PhysicalDevice) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::default();
        let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (i, family) in families.iter().enumerate() {
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i as u32);
            }
            if indices.is_complete() {
                break;
            }
        }
        indices
    }

    pub fn run(&self) {}
}

impl Drop for HelloTriangleApp {
    fn drop(&mut self) {
        unsafe {
            if let Some(m) = self.debug_messenger {
                let debug_utils = DebugUtils::new(&self.entry, &self.instance);
                debug_utils.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Default)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some()
    }
}

fn debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    vk::DebugUtilsMessengerCreateInfoEXT {
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::all(),
        pfn_user_callback: Some(debug_callback),
        ..Default::default()
    }
}

unsafe extern "system" fn debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let message = CStr::from_ptr((*p_callback_data).p_message);
    eprintln!("validation layer: {}", message.to_string_lossy());
    vk::FALSE
}
