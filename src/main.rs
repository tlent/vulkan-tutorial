use anyhow::Result;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

fn main() -> Result<()> {
    let app = HelloTriangleApp::new()?;
    app.run();
    Ok(())
}

struct HelloTriangleApp {
    event_loop: EventLoop<()>,
}

impl HelloTriangleApp {
    pub fn new() -> Result<Self> {
        let event_loop = Self::init_window()?;
        Self::init_vulkan();
        Ok(Self { event_loop })
    }

    fn init_window() -> Result<EventLoop<()>> {
        let event_loop = EventLoop::new();
        WindowBuilder::new()
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_title("Vulkan")
            .with_resizable(false)
            .build(&event_loop)?;
        Ok(event_loop)
    }

    fn init_vulkan() {}

    pub fn run(self) {
        self.event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                Event::LoopDestroyed => Self::cleanup(),
                Event::MainEventsCleared => {}
                _ => (),
            }
        });
    }

    fn cleanup() {}
}
