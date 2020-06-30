use std::cmp;
use std::ffi::{c_void, CStr, CString};
use std::mem;
use std::ptr;
use std::u32;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{vk, Device, Entry, Instance};
use lazy_static::lazy_static;
use memoffset::offset_of;
use nalgebra_glm as glm;
use winit::{
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, Window, WindowBuilder},
};

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

const VERT_SHADER_BYTES: &[u8] = include_bytes!("shaders/vert.spv");
const FRAG_SHADER_BYTES: &[u8] = include_bytes!("shaders/frag.spv");

#[cfg(debug_assertions)]
const VALIDATION_ENABLED: bool = true;
#[cfg(not(debug_assertions))]
const VALIDATION_ENABLED: bool = false;

lazy_static! {
    static ref VALIDATION_LAYERS: Vec<&'static CStr> =
        vec![CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap()];
    static ref DEVICE_EXTENSIONS: Vec<&'static CStr> = vec![Swapchain::name()];
    static ref VERTICES: Vec<Vertex> = vec![
        Vertex {
            position: glm::vec2(-0.5, -0.5),
            color: glm::vec3(1.0, 0.0, 0.0)
        },
        Vertex {
            position: glm::vec2(0.5, -0.5),
            color: glm::vec3(0.0, 1.0, 0.0)
        },
        Vertex {
            position: glm::vec2(0.5, 0.5),
            color: glm::vec3(0.0, 0.0, 1.0)
        },
        Vertex {
            position: glm::vec2(-0.5, 0.5),
            color: glm::vec3(1.0, 1.0, 1.0)
        },
    ];
    static ref INDICES: Vec<u16> = vec![0, 1, 2, 2, 3, 0];
}

fn main() {
    let (window, event_loop) = init_window();
    let mut app = HelloTriangleApp::new(&window);
    let mut pause_rendering = false;
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::KeyboardInput { input, .. } => match input.virtual_keycode {
                Some(VirtualKeyCode::Escape) => *control_flow = ControlFlow::Exit,
                _ => (),
            },
            WindowEvent::Resized(size) => {
                let size: (u32, u32) = size.into();
                if size == (0, 0) {
                    pause_rendering = true;
                    *control_flow = ControlFlow::Wait;
                } else {
                    pause_rendering = false;
                    *control_flow = ControlFlow::Poll;
                }
                app.window_resize(size)
            }
            _ => (),
        },
        Event::MainEventsCleared => {
            if pause_rendering {
                return;
            }
            app.draw_frame();
        }
        _ => (),
    });
}

fn init_window() -> (Window, EventLoop<()>) {
    let event_loop = EventLoop::new();
    let monitor = event_loop.primary_monitor();
    let window = WindowBuilder::new()
        .with_fullscreen(Some(Fullscreen::Borderless(monitor)))
        .with_title("Vulkan")
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();
    (window, event_loop)
}

struct HelloTriangleApp {
    window_size: (u32, u32),
    window_size_changed: bool,
    entry: Entry,
    instance: Instance,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    surface: vk::SurfaceKHR,
    surface_loader: Surface,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: QueueFamilyIndices,
    device: Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    swapchain_loader: Swapchain,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_imageviews: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<Option<vk::Fence>>,
    current_frame: u32,
}

impl HelloTriangleApp {
    pub fn new(window: &Window) -> Self {
        let window_size = window.inner_size().into();
        let entry = Entry::new().unwrap();
        let instance = Self::create_instance(&entry, window);
        let debug_messenger = if VALIDATION_ENABLED {
            Some(Self::setup_debug_messenger(&entry, &instance))
        } else {
            None
        };
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, window, None).unwrap() };
        let surface_loader = Surface::new(&entry, &instance);
        let (physical_device, queue_family_indices) =
            Self::select_physical_device(&instance, &surface_loader, surface);
        let (device, graphics_queue, present_queue) =
            Self::create_logical_device(&instance, physical_device, queue_family_indices);
        let swapchain_loader = Swapchain::new(&instance, &device);
        let (swapchain, swapchain_images, swapchain_image_format, swapchain_extent) =
            Self::create_swapchain(
                &swapchain_loader,
                &surface_loader,
                surface,
                physical_device,
                queue_family_indices,
                window_size,
                None,
            );
        let swapchain_imageviews =
            Self::create_imageviews(&device, &swapchain_images, swapchain_image_format);
        let render_pass = Self::create_render_pass(&device, swapchain_image_format);
        let (graphics_pipeline, pipeline_layout) =
            Self::create_graphics_pipeline(&device, render_pass, swapchain_extent);
        let swapchain_framebuffers = Self::create_framebuffers(
            &device,
            &swapchain_imageviews,
            render_pass,
            swapchain_extent,
        );
        let command_pool = Self::create_command_pool(&device, queue_family_indices);
        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
            &instance,
            &device,
            physical_device,
            graphics_queue,
            command_pool,
        );
        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
            &instance,
            &device,
            physical_device,
            graphics_queue,
            command_pool,
        );
        let command_buffers = Self::create_command_buffers(
            &device,
            command_pool,
            render_pass,
            &swapchain_framebuffers,
            vertex_buffer,
            index_buffer,
            swapchain_extent,
            graphics_pipeline,
        );
        let (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        ) = Self::create_sync_objects(&device, swapchain_images.len());
        Self {
            window_size,
            entry,
            instance,
            debug_messenger,
            surface,
            surface_loader,
            physical_device,
            queue_family_indices,
            device,
            graphics_queue,
            present_queue,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
            swapchain_imageviews,
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            swapchain_framebuffers,
            command_pool,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
            window_size_changed: false,
            current_frame: 0,
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
        if let Err(extension_name) =
            Self::check_instance_extension_support(entry, &required_extensions)
        {
            panic!(
                "Required extension {} is not supported.",
                extension_name.to_string_lossy()
            );
        }
        let extension_refs: Vec<_> = required_extensions.iter().map(|e| e.as_ptr()).collect();
        let mut instance_create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: extension_refs.len() as u32,
            pp_enabled_extension_names: extension_refs.as_ptr(),
            ..Default::default()
        };
        let layer_refs: Vec<_>;
        let debug_messenger_create_info;
        if VALIDATION_ENABLED {
            if let Err(layer_name) = Self::check_validation_layer_support(entry, &VALIDATION_LAYERS)
            {
                panic!(
                    "Validation layer {} is not supported.",
                    layer_name.to_string_lossy()
                );
            }
            layer_refs = VALIDATION_LAYERS.iter().map(|l| l.as_ptr()).collect();
            debug_messenger_create_info = get_debug_messenger_create_info();
            instance_create_info.enabled_layer_count = layer_refs.len() as u32;
            instance_create_info.pp_enabled_layer_names = layer_refs.as_ptr();
            instance_create_info.p_next = &debug_messenger_create_info
                as *const vk::DebugUtilsMessengerCreateInfoEXT
                as *const c_void;
        }
        let instance = unsafe { entry.create_instance(&instance_create_info, None).unwrap() };
        instance
    }

    fn check_instance_extension_support<'a>(
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
        let create_info = get_debug_messenger_create_info();
        let debug_utils = DebugUtils::new(entry, instance);
        unsafe {
            debug_utils
                .create_debug_utils_messenger(&create_info, None)
                .unwrap()
        }
    }

    fn select_physical_device(
        instance: &Instance,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, QueueFamilyIndices) {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        for d in devices {
            let indices = Self::find_queue_family_indices(instance, surface_loader, surface, d);
            if Self::is_device_suitable(instance, surface_loader, surface, d, indices) {
                return (d, indices);
            }
        }
        panic!("Failed to find a suitable GPU");
    }

    fn is_device_suitable(
        instance: &Instance,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
        indices: QueueFamilyIndices,
    ) -> bool {
        let supports_device_extensions =
            match Self::check_device_extension_support(instance, device) {
                Err(extension) => panic!(
                    "Required device extension not supported: {}",
                    extension.to_string_lossy()
                ),
                Ok(_) => true,
            };
        let is_swapchain_adequate = if supports_device_extensions {
            let swapchain_support = Self::query_swapchain_support(surface_loader, surface, device);
            !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
        } else {
            false
        };
        indices.is_complete() && supports_device_extensions && is_swapchain_adequate
    }

    fn find_queue_family_indices(
        instance: &Instance,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::default();
        let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (i, family) in families.iter().enumerate() {
            let index = i as u32;
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(index);
            }
            let present_support = unsafe {
                surface_loader
                    .get_physical_device_surface_support(device, index, surface)
                    .unwrap()
            };
            if present_support {
                indices.present_family = Some(index);
            }
            if indices.is_complete() {
                break;
            }
        }
        indices
    }

    fn check_device_extension_support(
        instance: &Instance,
        device: vk::PhysicalDevice,
    ) -> Result<(), &'static CStr> {
        let supported_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .unwrap()
        };
        let supported_extension_refs: Vec<_> = supported_extensions
            .iter()
            .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
            .collect();
        for e in DEVICE_EXTENSIONS.iter() {
            if !supported_extension_refs.contains(e) {
                return Err(e);
            }
        }
        Ok(())
    }

    fn create_logical_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_indices: QueueFamilyIndices,
    ) -> (Device, vk::Queue, vk::Queue) {
        let graphics_family_index = queue_family_indices.graphics_family.unwrap();
        let present_family_index = queue_family_indices.present_family.unwrap();
        let mut unique_queue_families = vec![graphics_family_index, present_family_index];
        unique_queue_families.sort_unstable();
        unique_queue_families.dedup();
        let queue_priority = 1.0;
        let queue_create_infos: Vec<_> = unique_queue_families
            .into_iter()
            .map(|index| vk::DeviceQueueCreateInfo {
                queue_family_index: index,
                queue_count: 1,
                p_queue_priorities: &queue_priority,
                ..Default::default()
            })
            .collect();
        let device_features = vk::PhysicalDeviceFeatures::default();
        let device_extension_refs: Vec<_> = DEVICE_EXTENSIONS.iter().map(|e| e.as_ptr()).collect();
        let mut device_create_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            p_enabled_features: &device_features,
            enabled_extension_count: device_extension_refs.len() as u32,
            pp_enabled_extension_names: device_extension_refs.as_ptr(),
            ..Default::default()
        };
        let layer_refs: Vec<_>;
        if VALIDATION_ENABLED {
            layer_refs = VALIDATION_LAYERS.iter().map(|l| l.as_ptr()).collect();
            device_create_info.enabled_layer_count = layer_refs.len() as u32;
            device_create_info.pp_enabled_layer_names = layer_refs.as_ptr();
        }
        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create logical device")
        };
        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };
        (device, graphics_queue, present_queue)
    }

    fn query_swapchain_support(
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> SwapchainSupportDetails {
        unsafe {
            let capabilities = surface_loader
                .get_physical_device_surface_capabilities(device, surface)
                .unwrap();
            let formats = surface_loader
                .get_physical_device_surface_formats(device, surface)
                .unwrap();
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(device, surface)
                .unwrap();
            SwapchainSupportDetails {
                capabilities,
                formats,
                present_modes,
            }
        }
    }

    fn choose_swap_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        available_formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .unwrap_or(available_formats[0])
    }

    fn choose_swap_present_mode(available_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        if available_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    fn choose_swap_extent(
        capabilities: vk::SurfaceCapabilitiesKHR,
        window_size: (u32, u32),
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }
        let (window_width, window_height) = window_size;
        let width = cmp::max(
            capabilities.min_image_extent.width,
            cmp::min(capabilities.max_image_extent.width, window_width),
        );
        let height = cmp::max(
            capabilities.min_image_extent.height,
            cmp::min(capabilities.max_image_extent.height, window_height),
        );
        vk::Extent2D { width, height }
    }

    fn create_swapchain(
        swapchain_loader: &Swapchain,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
        indices: QueueFamilyIndices,
        window_size: (u32, u32),
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> (vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D) {
        let swapchain_support = Self::query_swapchain_support(surface_loader, surface, device);

        let surface_format = Self::choose_swap_surface_format(&swapchain_support.formats);
        let present_mode = Self::choose_swap_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(swapchain_support.capabilities, window_size);
        let mut image_count = swapchain_support.capabilities.min_image_count + 1;
        let max_image_count = swapchain_support.capabilities.max_image_count;
        if max_image_count > 0 && image_count > max_image_count {
            image_count = max_image_count;
        }

        let mut create_info = vk::SwapchainCreateInfoKHR {
            surface: surface,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: present_mode,
            clipped: vk::TRUE,
            ..Default::default()
        };

        if let Some(sc) = old_swapchain {
            create_info.old_swapchain = sc;
        }

        let queue_family_indices: Vec<u32>;
        if indices.graphics_family != indices.present_family {
            queue_family_indices = vec![
                indices.graphics_family.unwrap(),
                indices.present_family.unwrap(),
            ];
            create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
            create_info.queue_family_index_count = queue_family_indices.len() as u32;
            create_info.p_queue_family_indices = queue_family_indices.as_ptr();
        }

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&create_info, None)
                .unwrap()
        };

        if let Some(sc) = old_swapchain {
            unsafe {
                swapchain_loader.destroy_swapchain(sc, None);
            }
        }

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };
        (swapchain, images, surface_format.format, extent)
    }

    fn create_imageviews(
        device: &Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Vec<vk::ImageView> {
        images
            .iter()
            .map(|&image| {
                let mut create_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    ..Default::default()
                };
                let subresource_range = &mut create_info.subresource_range;
                subresource_range.aspect_mask = vk::ImageAspectFlags::COLOR;
                subresource_range.base_mip_level = 0;
                subresource_range.level_count = 1;
                subresource_range.base_array_layer = 0;
                subresource_range.layer_count = 1;
                unsafe { device.create_image_view(&create_info, None).unwrap() }
            })
            .collect()
    }

    fn create_render_pass(device: &Device, swapchain_image_format: vk::Format) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            format: swapchain_image_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };
        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };
        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            ..Default::default()
        };

        let subpass_dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let create_info = vk::RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: &color_attachment,
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &subpass_dependency,
            ..Default::default()
        };

        unsafe { device.create_render_pass(&create_info, None).unwrap() }
    }

    fn create_graphics_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
        swapchain_extent: vk::Extent2D,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let vert_shader_module = Self::create_shader_module(device, VERT_SHADER_BYTES);
        let frag_shader_module = Self::create_shader_module(device, FRAG_SHADER_BYTES);

        let entry_point = CString::new("main").unwrap();
        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::VERTEX,
            module: vert_shader_module,
            p_name: entry_point.as_ptr(),
            ..Default::default()
        };
        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: frag_shader_module,
            p_name: entry_point.as_ptr(),
            ..Default::default()
        };
        let shader_stages = [vert_shader_stage_info, frag_shader_stage_info];

        let vertex_binding = Vertex::binding_description();
        let vertex_attributes = Vertex::attribute_descriptions();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &vertex_binding,
            vertex_attribute_description_count: vertex_attributes.len() as u32,
            p_vertex_attribute_descriptions: vertex_attributes.as_ptr(),
            ..Default::default()
        };
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        };
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
            ..Default::default()
        };
        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            line_width: 1.0,
            ..Default::default()
        };
        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            sample_shading_enable: vk::FALSE,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::all(),
            blend_enable: vk::FALSE,
            ..Default::default()
        };
        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        };

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo {
            stage_count: 2,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_color_blend_state: &color_blending,
            layout: pipeline_layout,
            render_pass: render_pass,
            subpass: 0,
            ..Default::default()
        };

        let pipelines;
        unsafe {
            pipelines = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .unwrap();
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        (pipelines[0], pipeline_layout)
    }

    fn create_shader_module(device: &Device, data: &[u8]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: data.len(),
            p_code: data.as_ptr() as *const u32,
            ..Default::default()
        };
        unsafe { device.create_shader_module(&create_info, None).unwrap() }
    }

    fn create_framebuffers(
        device: &Device,
        imageviews: &[vk::ImageView],
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        imageviews
            .iter()
            .map(|v| {
                let create_info = vk::FramebufferCreateInfo {
                    render_pass,
                    attachment_count: 1,
                    p_attachments: v,
                    width: extent.width,
                    height: extent.height,
                    layers: 1,
                    ..Default::default()
                };

                unsafe { device.create_framebuffer(&create_info, None).unwrap() }
            })
            .collect()
    }

    fn create_command_pool(device: &Device, indices: QueueFamilyIndices) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo {
            queue_family_index: indices.graphics_family.unwrap(),
            ..Default::default()
        };

        unsafe { device.create_command_pool(&create_info, None).unwrap() }
    }

    fn create_buffer(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: &Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = unsafe { device.create_buffer(&buffer_info, None).unwrap() };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type_index = Self::find_memory_type(
            instance,
            physical_device,
            mem_requirements.memory_type_bits,
            memory_properties,
        );
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_requirements.size,
            memory_type_index,
            ..Default::default()
        };
        let buffer_memory = unsafe { device.allocate_memory(&alloc_info, None).unwrap() };

        unsafe {
            device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();
        }
        (buffer, buffer_memory)
    }

    fn copy_buffer(
        device: &Device,
        transfer_queue: vk::Queue,
        command_pool: vk::CommandPool,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        let alloc_info = vk::CommandBufferAllocateInfo {
            level: vk::CommandBufferLevel::PRIMARY,
            command_pool,
            command_buffer_count: 1,
            ..Default::default()
        };

        unsafe {
            let command_buffer = device.allocate_command_buffers(&alloc_info).unwrap()[0];
            device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap();
            device.cmd_copy_buffer(
                command_buffer,
                src,
                dst,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                }],
            );
            device.end_command_buffer(command_buffer).unwrap();

            device
                .queue_submit(
                    transfer_queue,
                    &[vk::SubmitInfo {
                        command_buffer_count: 1,
                        p_command_buffers: &command_buffer,
                        ..Default::default()
                    }],
                    vk::Fence::null(),
                )
                .unwrap();
            device.queue_wait_idle(transfer_queue).unwrap();
            device.free_command_buffers(command_pool, &[command_buffer]);
        }
    }

    fn create_vertex_buffer(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        transfer_queue: vk::Queue,
        command_pool: vk::CommandPool,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = (VERTICES.len() * mem::size_of::<Vertex>()) as vk::DeviceSize;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::default(),
                )
                .unwrap();
            ptr::copy_nonoverlapping(VERTICES.as_ptr(), data as *mut Vertex, VERTICES.len());
            device.unmap_memory(staging_buffer_memory);
        }

        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self::copy_buffer(
            device,
            transfer_queue,
            command_pool,
            staging_buffer,
            vertex_buffer,
            buffer_size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer, vertex_buffer_memory)
    }

    fn create_index_buffer(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        transfer_queue: vk::Queue,
        command_pool: vk::CommandPool,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = (INDICES.len() * mem::size_of::<u16>()) as vk::DeviceSize;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::default(),
                )
                .unwrap();
            ptr::copy_nonoverlapping(INDICES.as_ptr(), data as *mut u16, INDICES.len());
            device.unmap_memory(staging_buffer_memory);
        }

        let (index_buffer, index_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self::copy_buffer(
            device,
            transfer_queue,
            command_pool,
            staging_buffer,
            index_buffer,
            buffer_size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (index_buffer, index_buffer_memory)
    }

    fn find_memory_type(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let mem_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        for (i, mem_type) in mem_properties.memory_types.iter().enumerate() {
            if type_filter & (1 << i) > 0 && mem_type.property_flags.contains(properties) {
                return i as u32;
            }
        }
        panic!("failed to find suitable memory type");
    }

    fn create_command_buffers(
        device: &Device,
        command_pool: vk::CommandPool,
        render_pass: vk::RenderPass,
        framebuffers: &[vk::Framebuffer],
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        extent: vk::Extent2D,
        pipeline: vk::Pipeline,
    ) -> Vec<vk::CommandBuffer> {
        let create_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: framebuffers.len() as u32,
            ..Default::default()
        };
        let command_buffers = unsafe { device.allocate_command_buffers(&create_info).unwrap() };
        for (&command_buffer, &framebuffer) in command_buffers.iter().zip(framebuffers) {
            let begin_info = vk::CommandBufferBeginInfo::default();
            let render_area = vk::Rect2D {
                extent,
                offset: vk::Offset2D { x: 0, y: 0 },
            };
            let clear_color = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            };
            let render_pass_info = vk::RenderPassBeginInfo {
                render_pass,
                framebuffer,
                render_area,
                clear_value_count: 1,
                p_clear_values: &clear_color,
                ..Default::default()
            };
            unsafe {
                device
                    .begin_command_buffer(command_buffer, &begin_info)
                    .unwrap();
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    index_buffer,
                    0,
                    vk::IndexType::UINT16,
                );
                device.cmd_draw_indexed(command_buffer, INDICES.len() as u32, 1, 0, 0, 0);
                device.cmd_end_render_pass(command_buffer);
                device.end_command_buffer(command_buffer).unwrap();
            }
        }
        command_buffers
    }

    fn create_sync_objects(
        device: &Device,
        image_count: usize,
    ) -> (
        Vec<vk::Semaphore>,
        Vec<vk::Semaphore>,
        Vec<vk::Fence>,
        Vec<Option<vk::Fence>>,
    ) {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        let mut image_available_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
        let mut render_finished_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
        let mut in_flight_fences = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let semaphore = unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };
            image_available_semaphores.push(semaphore);
            let semaphore = unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };
            render_finished_semaphores.push(semaphore);
            let fence = unsafe { device.create_fence(&fence_info, None).unwrap() };
            in_flight_fences.push(fence);
        }
        let images_in_flight = vec![None; image_count];
        (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        )
    }

    pub fn draw_frame(&mut self) {
        let current_frame = self.current_frame as usize;
        let current_in_flight_fence = self.in_flight_fences[current_frame];
        unsafe {
            self.device
                .wait_for_fences(&[current_in_flight_fence], true, std::u64::MAX)
                .unwrap();
        }
        let result = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                self.image_available_semaphores[current_frame],
                vk::Fence::null(),
            )
        };
        let image_index = match result {
            Ok((i, false)) => i,
            Ok((_, true)) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain();
                return;
            }
            Err(e) => panic!(e),
        };

        let current_image_fence = &mut self.images_in_flight[image_index as usize];
        if let Some(fence) = *current_image_fence {
            unsafe {
                self.device
                    .wait_for_fences(&[fence], true, std::u64::MAX)
                    .unwrap();
            }
        }

        *current_image_fence = Some(current_in_flight_fence);

        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.image_available_semaphores[current_frame],
            p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[image_index as usize],
            signal_semaphore_count: 1,
            p_signal_semaphores: &self.render_finished_semaphores[current_frame],
            ..Default::default()
        };
        unsafe {
            self.device
                .reset_fences(&[current_in_flight_fence])
                .unwrap();
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], current_in_flight_fence)
                .unwrap();
        }

        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.render_finished_semaphores[current_frame],
            swapchain_count: 1,
            p_swapchains: &self.swapchain,
            p_image_indices: &image_index,
            ..Default::default()
        };

        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.graphics_queue, &present_info)
        };

        match result.map(|is_suboptimal| is_suboptimal || self.window_size_changed) {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate_swapchain(),
            Err(e) => panic!(e),
            Ok(false) => (),
        };

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn recreate_swapchain(&mut self) {
        unsafe { self.device.device_wait_idle().unwrap() };

        self.window_size_changed = false;
        self.cleanup_swapchain();
        let (swapchain, swapchain_images, swapchain_image_format, swapchain_extent) =
            Self::create_swapchain(
                &self.swapchain_loader,
                &self.surface_loader,
                self.surface,
                self.physical_device,
                self.queue_family_indices,
                self.window_size,
                Some(self.swapchain),
            );
        let swapchain_imageviews =
            Self::create_imageviews(&self.device, &swapchain_images, swapchain_image_format);
        let render_pass = Self::create_render_pass(&self.device, swapchain_image_format);
        let (graphics_pipeline, pipeline_layout) =
            Self::create_graphics_pipeline(&self.device, render_pass, swapchain_extent);
        let swapchain_framebuffers = Self::create_framebuffers(
            &self.device,
            &swapchain_imageviews,
            render_pass,
            swapchain_extent,
        );
        let command_buffers = Self::create_command_buffers(
            &self.device,
            self.command_pool,
            render_pass,
            &swapchain_framebuffers,
            self.vertex_buffer,
            self.index_buffer,
            swapchain_extent,
            graphics_pipeline,
        );
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_format = swapchain_image_format;
        self.swapchain_extent = swapchain_extent;
        self.swapchain_imageviews = swapchain_imageviews;
        self.render_pass = render_pass;
        self.graphics_pipeline = graphics_pipeline;
        self.pipeline_layout = pipeline_layout;
        self.swapchain_framebuffers = swapchain_framebuffers;
        self.command_buffers = command_buffers;
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            for &f in &self.swapchain_framebuffers {
                self.device.destroy_framebuffer(f, None);
            }
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for &v in &self.swapchain_imageviews {
                self.device.destroy_image_view(v, None);
            }
        }
    }

    pub fn window_resize(&mut self, window_size: (u32, u32)) {
        self.window_size = window_size;
        self.window_size_changed = true;
    }
}

impl Drop for HelloTriangleApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            let semaphores = self
                .image_available_semaphores
                .iter()
                .chain(&self.render_finished_semaphores);
            for &semaphore in semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &fence in &self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
            self.cleanup_swapchain();
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);
            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some(m) = self.debug_messenger {
                let debug_utils = DebugUtils::new(&self.entry, &self.instance);
                debug_utils.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Default, Clone, Copy)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

#[derive(Debug)]
struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

fn get_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
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

#[derive(Debug)]
struct Vertex {
    position: glm::Vec2,
    color: glm::Vec3,
}

impl Vertex {
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    fn attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Vertex, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, color) as u32,
            },
        ]
    }
}
