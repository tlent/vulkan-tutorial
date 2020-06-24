use std::cmp;
use std::ffi::{c_void, CStr, CString};
use std::u32;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{vk, Device, Entry, Instance};
use lazy_static::lazy_static;
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

const VERT_SHADER_BYTES: &[u8] = include_bytes!("shaders/vert.spv");
const FRAG_SHADER_BYTES: &[u8] = include_bytes!("shaders/frag.spv");

lazy_static! {
    static ref VALIDATION_LAYERS: Vec<&'static CStr> =
        vec![CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap()];
    static ref DEVICE_EXTENSIONS: Vec<&'static CStr> = vec![Swapchain::name()];
}

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
    surface: vk::SurfaceKHR,
    surface_loader: Surface,
    physical_device: vk::PhysicalDevice,
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
        Self {
            entry,
            instance,
            debug_messenger,
            surface,
            surface_loader,
            physical_device,
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
                f.format == vk::Format::B8G8R8_SRGB
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

    fn choose_swap_extent(capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }
        let width = cmp::max(
            capabilities.min_image_extent.width,
            cmp::min(capabilities.max_image_extent.width, WINDOW_WIDTH),
        );
        let height = cmp::max(
            capabilities.min_image_extent.height,
            cmp::min(capabilities.max_image_extent.height, WINDOW_HEIGHT),
        );
        vk::Extent2D { width, height }
    }

    fn create_swapchain(
        swapchain_loader: &Swapchain,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
        indices: QueueFamilyIndices,
    ) -> (vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D) {
        let swapchain_support = Self::query_swapchain_support(surface_loader, surface, device);

        let surface_format = Self::choose_swap_surface_format(&swapchain_support.formats);
        let present_mode = Self::choose_swap_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(swapchain_support.capabilities);
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

        let create_info = vk::RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: &color_attachment,
            subpass_count: 1,
            p_subpasses: &subpass,
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

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
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

    pub fn run(&self) {}
}

impl Drop for HelloTriangleApp {
    fn drop(&mut self) {
        unsafe {
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
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
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
