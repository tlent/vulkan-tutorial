use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;

use ash::extensions::khr::Surface;
use ash::version::{EntryV1_0, InstanceV1_0};
use ash::{vk, Entry, Instance};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
use ash::extensions::khr::XlibSurface;
#[cfg(target_os = "macos")]
use ash::extensions::mvk::MacOSSurface;

#[cfg(target_os = "macos")]
use cocoa::appkit::{NSView, NSWindow};
#[cfg(target_os = "macos")]
use cocoa::base::id as cocoa_id;
#[cfg(target_os = "macos")]
use metal::CoreAnimationLayer;
#[cfg(target_os = "macos")]
use objc::runtime::YES;
#[cfg(target_os = "macos")]
use std::mem;

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

fn main() {
    let (window, event_loop) = init_window();
    let app = HelloTriangleApp::new(window);
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
    window: Window,
}

impl HelloTriangleApp {
    pub fn new(window: Window) -> Self {
        let (entry, instance) = Self::init_vulkan();
        Self {
            entry,
            instance,
            window,
        }
    }

    fn init_vulkan() -> (Entry, Instance) {
        let entry = Entry::new().unwrap();
        let instance = Self::create_instance(&entry);
        (entry, instance)
    }

    fn create_instance(entry: &Entry) -> Instance {
        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_application_name: CString::new("Hello Triangle").unwrap().as_ptr(),
            application_version: vk::make_version(1, 0, 0),
            p_engine_name: CString::new("No Engine").unwrap().as_ptr(),
            engine_version: vk::make_version(1, 0, 0),
            api_version: vk::make_version(1, 0, 0),
            ..Default::default()
        };
        let required_extensions = required_extensions();
        let supported_extensions: Vec<_> = entry
            .enumerate_instance_extension_properties()
            .unwrap()
            .iter()
            .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
            .collect();
        for &e in required_extensions.iter() {
            let extension_name = unsafe { CStr::from_ptr(e) };
            if !supported_extensions.contains(&extension_name) {
                panic!(
                    "Required extension {} is not supported.",
                    extension_name.to_str().unwrap()
                );
            }
        }
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: required_extensions.len() as u32,
            pp_enabled_extension_names: required_extensions.as_ptr(),
            ..Default::default()
        };
        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };
        instance
    }

    pub fn run(&self) {}
}

impl Drop for HelloTriangleApp {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
fn required_extensions() -> Vec<*const c_char> {
    vec![Surface::name().as_ptr(), XlibSurface::name().as_ptr()]
}

#[cfg(target_os = "macos")]
fn required_extensions() -> Vec<*const c_char> {
    vec![Surface::name().as_ptr(), MacOSSurface::name().as_ptr()]
}

#[cfg(all(windows))]
fn required_extensions() -> Vec<*const c_char> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use winit::platform::unix::WindowExtUnix;
    let x11_display = window.xlib_display().unwrap();
    let x11_window = window.xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
        .window(x11_window)
        .dpy(x11_display as *mut vk::Display);

    let xlib_surface_loader = XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

#[cfg(target_os = "macos")]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::ptr;
    use winit::platform::macos::WindowExtMacOS;

    let wnd: cocoa_id = mem::transmute(window.ns_window());

    let layer = CoreAnimationLayer::new();

    layer.set_edge_antialiasing_mask(0);
    layer.set_presents_with_transaction(false);
    layer.remove_all_animations();

    let view = wnd.contentView();

    layer.set_contents_scale(view.backingScaleFactor());
    view.setLayer(mem::transmute(layer.as_ref()));
    view.setWantsLayer(YES);

    let create_info = vk::MacOSSurfaceCreateInfoMVK {
        s_type: vk::StructureType::MACOS_SURFACE_CREATE_INFO_M,
        p_next: ptr::null(),
        flags: Default::default(),
        p_view: window.ns_view() as *const c_void,
    };

    let macos_surface_loader = MacOSSurface::new(entry, instance);
    macos_surface_loader.create_mac_os_surface_mvk(&create_info, None)
}

#[cfg(target_os = "windows")]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::ptr;
    use winapi::shared::windef::HWND;
    use winapi::um::libloaderapi::GetModuleHandleW;
    use winit::platform::windows::WindowExtWindows;

    let hwnd = window.hwnd() as HWND;
    let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
        s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        hinstance,
        hwnd: hwnd as *const c_void,
    };
    let win32_surface_loader = Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}
