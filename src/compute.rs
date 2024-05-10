use {
    crate::block_on,
    glam::{UVec2, Vec2, Vec4},
    gpgpu::{
        BufOps, DescriptorSet, GpuBuffer, GpuBufferUsage, GpuUniformBuffer, Kernel, Program, Shader,
    },
    image::{io::Reader, RgbaImage},
    rand::Rng,
    std::{io::Cursor, thread::JoinHandle},
    wgpu::{
        util, util::DeviceExt, Backends, Color, CommandEncoderDescriptor, CompositeAlphaMode,
        DeviceDescriptor, Instance, InstanceDescriptor, LoadOp, PowerPreference, PresentMode,
        RenderPassColorAttachment, RenderPassDescriptor, RequestAdapterOptions, StoreOp,
        SurfaceConfiguration, SurfaceTargetUnsafe, TextureUsages, TextureViewDescriptor,
    },
    winit::{
        dpi::PhysicalSize,
        raw_window_handle::{HasDisplayHandle, HasWindowHandle},
        window::Window,
    },
};

lazy_static::lazy_static! {
    static ref FW: gpgpu::Framework = gpgpu::Framework::default();
    pub static ref BLUE_TEXTURE: RgbaImage = {
        Reader::new(Cursor::new(BLUE_NOISE)).with_guessed_format().unwrap().decode().unwrap().into_rgba8()
    };
}

#[allow(non_upper_case_globals)]
mod shaders {
    pub const main_cs: &str = "main_cs";
}

struct RenderPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    render_buffer: wgpu::Buffer,
}

impl RenderPipeline {
    fn prepare(&self, que: &wgpu::Queue, frame: &[f32], width: u32, height: u32) {
        que.write_buffer(&self.render_buffer, 0, bytemuck::cast_slice(frame));
        que.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[width, height]));
    }

    fn paint<'rpass>(&'rpass self, rpass: &mut wgpu::RenderPass<'rpass>) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..6, 0..1);
    }

    fn new(
        dev: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> RenderPipeline {
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("k/post.wgsl").into()),
        });

        let bind_group_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let uniform_buffer = dev.create_buffer_init(&util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0.0, 0.0, 0.0]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        let render_buffer = dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (width * height * 3 * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: render_buffer.as_entire_binding() },
            ],
        });

        RenderPipeline { pipeline, bind_group, uniform_buffer, render_buffer }
    }
}

pub struct Wgpu<'a> {
    dev: wgpu::Device,
    que: wgpu::Queue,
    surface: wgpu::Surface<'a>,
    format: wgpu::TextureFormat,

    pipeline: RenderPipeline,
    compute_handle: Option<JoinHandle<()>>,
}

impl<'a> Wgpu<'a> {
    pub fn init(window: &Window) -> Self {
        let instance = Instance::new(InstanceDescriptor {
            dx12_shader_compiler: util::dx12_shader_compiler_from_env().unwrap_or_default(),
            backends: Backends::PRIMARY,
            ..Default::default()
        });
        let surface = unsafe {
            instance.create_surface_unsafe(SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: window.display_handle().unwrap().as_raw(),
                raw_window_handle: window.window_handle().unwrap().as_raw(),
            })
        }
        .unwrap();
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to creator wgpu adapter.");

        let (dev, que) =
            block_on(adapter.request_device(&DeviceDescriptor { ..Default::default() }, None))
                .expect("Failed to creator wgpu device.");

        let size = window.inner_size();
        let format = surface.get_capabilities(&adapter).formats[0];
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&dev, &config);

        let size = window.inner_size();
        let pipeline = RenderPipeline::new(&dev, format, size.width, size.height);
        Wgpu { dev, que, surface, format, pipeline, compute_handle: None }
    }

    pub fn redraw(&self, buf: &[f32]) {
        let Ok(frame) = self.surface.get_current_texture() else { return };

        self.pipeline.prepare(&self.que, &buf, 1280, 720);

        let mut command_encoder =
            self.dev.create_command_encoder(&CommandEncoderDescriptor::default());
        let view = &frame.texture.create_view(&TextureViewDescriptor::default());

        {
            let mut pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.pipeline.paint(&mut pass);
        }
        self.que.submit(Some(command_encoder.finish()));

        frame.present();
    }

    pub fn start_render(&mut self, size: PhysicalSize<u32>, continue_previous: bool) {
        if self.compute_handle.is_some() {
            self.stop_render();
        }

        if !continue_previous {
            self.pipeline = RenderPipeline::new(&self.dev, self.format, size.width, size.height);
        }
    }

    fn stop_render(&mut self) {
        // if let Some(handle) = self.compute_handle.take() {
        //     rx.send(()).unwrap();
        //     handle.join().unwrap();
        //     self.compute_handle = None;
        // }
    }
}

use shared::TracingConfig;

const KERNEL: &[u8] = include_bytes!("k.gen/simple");
const BLUE_NOISE: &[u8] = include_bytes!("k/blue.png");

pub struct Tracing {
    pub frame: Vec<f32>,
    pub config: TracingConfig,
    pub samples: usize,
}

impl Tracing {
    pub fn frame(width: u32, height: u32) -> Vec<f32> {
        vec![0.0; width as usize * height as usize * 3]
    }

    pub fn new(config: TracingConfig) -> Self {
        Self { frame: Self::frame(config.width, config.height), config, samples: 0 }
    }
}

struct PathTracing<'fw>(Kernel<'fw>);

impl<'fw> PathTracing<'fw> {
    fn new(
        config_buf: &GpuUniformBuffer<'fw, TracingConfig>,
        rng_buffer: &GpuBuffer<'fw, UVec2>,
        output_buf: &GpuBuffer<'fw, Vec4>,
    ) -> Self {
        let shader = Shader::from_spirv_bytes(&FW, KERNEL, Some("compute"));
        // let sampler = Sampler::new(&FW, SamplerWrapMode::ClampToEdge, SamplerFilterMode::Linear);
        let bindings = DescriptorSet::default()
            .bind_uniform_buffer(config_buf)
            .bind_buffer(rng_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(output_buf, GpuBufferUsage::ReadWrite);
        Self(Kernel::new(&FW, Program::new(&shader, "main_cs").add_descriptor_set(bindings)))
    }
}

pub fn trace_gpu(mut state: &mut Tracing, crv: Vec2) -> &[f32] {
    let TracingConfig { width, height, .. } = state.config;

    let pixel_count = (width * height) as usize;

    let mut rng = rand::thread_rng();
    let mut blue: Vec<UVec2> = vec![UVec2::ZERO; pixel_count];
    let mut uniform: Vec<UVec2> = vec![UVec2::ZERO; pixel_count];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let pixel = BLUE_TEXTURE.get_pixel(x % BLUE_TEXTURE.width(), y % BLUE_TEXTURE.height())
                [0] as f32
                / 255.0;
            blue[idx].x = 0;
            blue[idx].y = (pixel * 4294967295.0) as u32;
            uniform[idx].x = Rng::gen(&mut rng);
        }
    }

    let samples = state.samples as f32;
    let raw_buf = state
        .frame
        .chunks(3)
        .map(|c| Vec4::new(c[0], c[1], c[2], 1.0) * samples)
        .collect::<Vec<_>>();

    let config_buf = GpuUniformBuffer::from_slice(&FW, &[state.config]);
    let rng_buf = GpuBuffer::from_slice(&FW, &uniform);
    let output_buf = GpuBuffer::from_slice(&FW, &raw_buf);
    let rt = PathTracing::new(&config_buf, &rng_buf, &output_buf);

    let mut image_buf_raw: Vec<Vec4> = vec![Vec4::ZERO; pixel_count];
    let mut image_buf: Vec<f32> = vec![0.0; pixel_count * 3];

    rt.0.enqueue(width.div_ceil(8), height.div_ceil(8), 1);
    FW.poll_blocking();

    state.samples += 1;

    let _ = output_buf.read_blocking(&mut image_buf_raw[..]);
    for (i, col) in image_buf_raw.iter().enumerate() {
        image_buf[i * 3] = col.x / (samples + 1.0);
        image_buf[i * 3 + 1] = col.y / (samples + 1.0);
        image_buf[i * 3 + 2] = col.z / (samples + 1.0);
    }

    state.frame.copy_from_slice(image_buf.as_slice());
    &state.frame
}
