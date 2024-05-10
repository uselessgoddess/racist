#![feature(build_hasher_simple_hash_one)]
#![feature(sync_unsafe_cell)]

mod block;
mod compute;

pub(crate) use block::block_on;
use {
    crate::compute::Tracing,
    compute::Wgpu,
    glam::Vec2,
    parking_lot::Mutex,
    shared::TracingConfig,
    std::{sync::Arc, thread, time::Instant},
    winit::{
        application::ApplicationHandler,
        dpi::{PhysicalPosition, PhysicalSize},
        event::WindowEvent,
        event_loop::{ActiveEventLoop, EventLoop},
        raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
        window::{Window, WindowAttributes, WindowId},
    },
};

struct App<'a> {
    window: &'a Window,
    req: Request,
    crs: PhysicalPosition<f64>,
    crv: Arc<Mutex<Vec2>>,
}

impl<'a> App<'a> {
    pub fn new(window: &'a Window) -> Self {
        Self {
            window,
            req: Request { close: false },
            crs: Default::default(),
            crv: Arc::new(Mutex::new(Vec2::ZERO)),
        }
    }

    pub fn redraw_frame(&mut self) {}

    pub fn start_render(&mut self, continue_previous: bool) {
        // self.wgpu.start_render(self.window.inner_size(), continue_previous)
    }
}

struct Request {
    close: bool,
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        self.start_render(false);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::RedrawRequested => self.redraw_frame(),
            WindowEvent::CloseRequested => {
                println!("start render");
                self.start_render(true);
                // self.req.close = true;
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.req.close {
            event_loop.exit();
        }
    }
}

fn main() {
    let event_loop = EventLoop::<()>::with_user_event().build().unwrap();
    let window = event_loop
        .create_window(
            WindowAttributes::default()
                .with_title("racist")
                .with_inner_size(PhysicalSize { width: 1280, height: 720 }),
        )
        .unwrap();

    let mut app = App::new(&window);
    let wgpu = Wgpu::init(app.window);

    let crv = app.crv.clone();
    let PhysicalSize { width, height } = window.inner_size();
    thread::spawn(move || {
        let mut state = Tracing::new(TracingConfig { width, height });
        loop {
            let instant = Instant::now();
            {
                wgpu.redraw(&compute::trace_gpu(&mut state, *crv.lock()));
            }
            println!("frame elapsed: {:?}", instant.elapsed());
        }
    });

    event_loop.run_app(&mut app).unwrap();
}
