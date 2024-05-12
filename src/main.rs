#![feature(build_hasher_simple_hash_one)]
#![feature(sync_unsafe_cell)]

mod atlas;
mod block;
mod bvh;
mod compute;
mod light;
mod scene;

pub(crate) use block::block_on;
use {
    crate::{compute::Tracing, scene::World},
    compute::Wgpu,
    glam::{Mat3, Vec3},
    parking_lot::Mutex,
    shared::TracingConfig,
    std::{sync::Arc, thread, time::Instant},
    winit::{
        application::ApplicationHandler,
        dpi::{LogicalPosition, PhysicalPosition, PhysicalSize},
        event::WindowEvent,
        event_loop::{ActiveEventLoop, EventLoop},
        keyboard::{Key, KeyCode, PhysicalKey},
        raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
        window::{Window, WindowAttributes, WindowId},
    },
};

struct App<'a> {
    window: &'a Window,
    req: Request,
    config: Arc<Mutex<TracingConfig>>,
}

impl<'a> App<'a> {
    pub fn new(window: &'a Window) -> Self {
        let PhysicalSize { width, height } = window.inner_size();
        Self {
            window,
            req: Request { close: false },
            config: Arc::new(Mutex::new(TracingConfig { width, height, ..TracingConfig::soft() })),
        }
    }

    pub fn redraw_frame(&mut self) {}

    pub fn start_render(&mut self, continue_previous: bool) {
        // self.wgpu.start_render(self.window.inner_size(), continue_previous)
    }

    pub fn handle_mouse(&mut self, delta: (f64, f64)) {
        let mut config = self.config.lock();

        config.cam_rot.x += delta.1 as f32 * 0.005;
        config.cam_rot.y += delta.0 as f32 * 0.005;
    }

    pub fn handle_input(&mut self, key: PhysicalKey, ctrl: Key) {
        let mut config = self.config.lock();

        let mut forward = Vec3::new(0.0, 0.0, 1.0);
        let mut right = Vec3::new(1.0, 0.0, 0.0);
        let euler_mat =
            Mat3::from_rotation_y(config.cam_rot.y) * Mat3::from_rotation_x(config.cam_rot.x);
        forward = euler_mat * forward;
        right = euler_mat * right;
        let speed = 0.1;

        if matches!(key, PhysicalKey::Code(KeyCode::KeyW)) {
            config.cam_pos += forward.extend(0.0) * speed;
        }
        if matches!(key, PhysicalKey::Code(KeyCode::KeyS)) {
            config.cam_pos -= forward.extend(0.0) * speed;
        }
        if matches!(key, PhysicalKey::Code(KeyCode::KeyD)) {
            config.cam_pos += right.extend(0.0) * speed;
        }
        if matches!(key, PhysicalKey::Code(KeyCode::KeyA)) {
            config.cam_pos -= right.extend(0.0) * speed;
        }

        println!("position: {:?}", config.cam_pos);
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
            WindowEvent::CursorMoved { position: PhysicalPosition { x, y }, .. } => {
                // Set mouse position to center of screen
                let size = self.window.inner_size();
                let center =
                    LogicalPosition::new(size.width as f32 / 2.0, size.height as f32 / 2.0);
                let _ = self.window.set_cursor_position(center);

                let sensitivity = 0.7;
                let delta =
                    ((x - center.x as f64) * sensitivity, (y - center.y as f64) * sensitivity);

                self.handle_mouse(delta);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    self.handle_input(event.physical_key, event.logical_key);
                }
            }
            WindowEvent::CloseRequested => {
                self.req.close = true;
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
    let (width, height) = (1400, 1400);
    let window = event_loop
        .create_window(
            WindowAttributes::default()
                .with_title("racist")
                .with_inner_size(PhysicalSize { width, height }),
        )
        .unwrap();

    let mut app = App::new(&window);
    let wgpu = Wgpu::init(app.window);

    let world = World::from_path("PBRTest.glb").unwrap().into_gpu();

    let config = app.config.clone();
    let mut state = Tracing::new(*config.lock());
    thread::spawn(move || loop {
        let update = *config.clone().lock();
        if update.cam_rot != state.config.cam_rot || update.cam_pos != state.config.cam_pos {
            state.samples = 0;
            state.frame.fill(0.0);
        }
        state.config = update;
        wgpu.redraw(&compute::trace_gpu(&mut state, &world), width, height);
    });

    event_loop.run_app(&mut app).unwrap();
}
