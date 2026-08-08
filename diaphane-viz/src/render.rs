//! Volume rendering of a live simulation.
//!
//! The fragment shader binds the solver's own field buffers. Nothing is
//! copied, converted, or staged between the compute kernel writing a field and
//! the pixel that displays it.

use crate::camera::Camera;
use blade_graphics as gpu;
use diaphane::{Extent, gpu::Simulation};
use std::{fs, io, mem, path::Path, sync::Arc};

/// Which quantity the volume shows.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ViewMode {
    /// Electric and magnetic energy density in opposing hues. The default,
    /// because it is the view in which the two fields visibly trade energy.
    #[default]
    Energy,
    /// Signed `Ez` through a diverging colormap.
    Electric,
    /// Signed `Hz`.
    Magnetic,
    /// Total energy density, monochrome.
    Magnitude,
}

impl ViewMode {
    pub const ALL: [Self; 4] = [
        Self::Energy,
        Self::Electric,
        Self::Magnetic,
        Self::Magnitude,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Energy => "energy split",
            Self::Electric => "Ez",
            Self::Magnetic => "Hz",
            Self::Magnitude => "total energy",
        }
    }

    fn code(self) -> f32 {
        match self {
            Self::Energy => 0.0,
            Self::Electric => 1.0,
            Self::Magnetic => 2.0,
            Self::Magnitude => 3.0,
        }
    }

    fn is_signed(self) -> bool {
        match self {
            Self::Electric | Self::Magnetic => true,
            Self::Energy | Self::Magnitude => false,
        }
    }
}

/// Everything the display does that the physics does not.
#[derive(Clone, Copy, Debug)]
pub struct ViewSettings {
    pub camera: Camera,
    pub mode: ViewMode,
    /// Multiplies the auto-ranged scale. 1.0 puts the loudest cell at the top
    /// of the range.
    pub gain: f32,
    /// Strength of the signed-log compression; 0 is linear.
    pub log_strength: f32,
}

impl ViewSettings {
    pub fn new() -> Self {
        Self {
            camera: Camera::new(),
            mode: ViewMode::default(),
            gain: 1.0,
            // Enough compression that a weak scattered field is visible in
            // the same frame as the source that produced it, which linear
            // scaling hides completely. Much more than this and the noise
            // floor comes up with it.
            log_strength: 6.0,
        }
    }
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewParams {
    /// `nx, ny, nz, cell_count`.
    extent: [u32; 4],
    /// Camera position in cells, then the ray-march step in cells.
    origin: [f32; 4],
    /// Camera right scaled by aspect and field of view, then exposure.
    right: [f32; 4],
    /// Camera up scaled by field of view, then the signed-log strength.
    up: [f32; 4],
    /// Camera forward, then the view mode.
    forward: [f32; 4],
    /// Reciprocal of the reference path length, then padding.
    tone: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct VolumeData {
    electric: gpu::BufferPiece,
    magnetic: gpu::BufferPiece,
    peak: gpu::BufferPiece,
    view: ViewParams,
}

/// Draws a [`Simulation`] into a colour target.
pub struct Renderer {
    context: Arc<gpu::Context>,
    volume: gpu::RenderPipeline,
    measure: gpu::ComputePipeline,
    peak: gpu::Buffer,
    /// Auto-ranged peak energy density, smoothed across frames.
    scale: f32,
}

impl Renderer {
    pub fn new(context: Arc<gpu::Context>, format: gpu::TextureFormat) -> Self {
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("shaders/volume.wgsl"),
            naga_module: None,
        });
        shader.check_struct_size::<ViewParams>();
        let layout = <VolumeData as gpu::ShaderData>::layout();

        let volume = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "volume",
            data_layouts: &[&layout],
            vertex: shader.at("main_vs"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            fragment: Some(shader.at("main_fs")),
            color_targets: &[format.into()],
            multisample_state: gpu::MultisampleState::default(),
        });
        let measure = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "measure peak",
            data_layouts: &[&layout],
            compute: shader.at("measure_peak"),
        });
        // Host-visible so the reduction can be read without a staging copy;
        // it is four bytes, and the read is a frame behind the write.
        let peak = context.create_buffer(gpu::BufferDesc {
            name: "field peak",
            size: mem::size_of::<u32>() as u64,
            memory: gpu::Memory::Shared,
        });

        Self {
            context,
            volume,
            measure,
            peak,
            scale: 0.0,
        }
    }

    /// Reads the previous frame's reduction and folds it into the running
    /// scale.
    ///
    /// Rises quickly and falls slowly. Without that asymmetry the display
    /// flickers as a pulse's amplitude swings: every dip in the peak would be
    /// answered by an immediate brightening.
    fn update_scale(&mut self) {
        // SAFETY: `peak` is host-visible, four bytes long, and the submission
        // that wrote it has completed before this runs.
        let measured = f32::from_bits(unsafe { self.peak.data().cast::<u32>().read_unaligned() });
        if !measured.is_finite() || measured <= 0.0 {
            return;
        }
        if self.scale == 0.0 {
            self.scale = measured;
            return;
        }
        let rate = if measured > self.scale { 0.4 } else { 0.02 };
        self.scale += rate * (measured - self.scale);
    }

    fn params(
        &self,
        extent: Extent,
        settings: &ViewSettings,
        target: gpu::Extent,
        march: f32,
    ) -> ViewParams {
        let aspect = target.width as f32 / target.height.max(1) as f32;
        let basis = settings.camera.basis(extent, aspect);
        // The reduction measures energy density; the signed views show a field
        // amplitude, whose scale is the square root of it.
        let reference = self.scale.max(f32::MIN_POSITIVE);
        let exposure = if settings.mode.is_signed() {
            settings.gain / (2.0 * reference).sqrt()
        } else {
            settings.gain / reference
        };
        ViewParams {
            extent: [extent.x, extent.y, extent.z, extent.total() as u32],
            origin: [
                basis.position[0],
                basis.position[1],
                basis.position[2],
                march,
            ],
            right: [basis.right[0], basis.right[1], basis.right[2], exposure],
            up: [basis.up[0], basis.up[1], basis.up[2], settings.log_strength],
            forward: [
                basis.forward[0],
                basis.forward[1],
                basis.forward[2],
                settings.mode.code(),
            ],
            tone: [1.0 / reference_path(extent), 0.0, 0.0, 0.0],
        }
    }

    /// Encodes the auto-range reduction on its own.
    ///
    /// The reduction is read back a frame late, so the very first frame would
    /// otherwise be drawn with no scale at all — which, since exposure is the
    /// reciprocal of it, means a completely white image. Callers run this once
    /// and submit before their first [`Self::draw`].
    pub fn measure(&mut self, encoder: &mut gpu::CommandEncoder, simulation: &Simulation) {
        let extent = simulation.grid().extent;
        let data = VolumeData {
            electric: simulation.electric_buffer(),
            magnetic: simulation.magnetic_buffer(),
            peak: self.peak.into(),
            view: ViewParams {
                extent: [extent.x, extent.y, extent.z, extent.total() as u32],
                ..Default::default()
            },
        };
        {
            let mut pass = encoder.transfer("clear peak");
            pass.fill_buffer(self.peak.into(), mem::size_of::<u32>() as u64, 0);
        }
        let mut pass = encoder.compute("measure peak");
        let mut pipeline = pass.with(&self.measure);
        pipeline.bind(0, &data);
        pipeline.dispatch(self.measure.get_dispatch_for(gpu::Extent {
            width: extent.x,
            height: extent.y,
            depth: extent.z,
        }));
    }

    /// Encodes the reduction and the volume pass. The caller submits.
    pub fn draw(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        target: gpu::TextureView,
        size: gpu::Extent,
        simulation: &Simulation,
        settings: &ViewSettings,
    ) {
        self.update_scale();
        self.measure(encoder, simulation);

        let extent = simulation.grid().extent;
        // One sample per cell along the ray is enough at the resolutions the
        // solver is usable at; the field varies over tens of cells.
        let march = 1.0;
        let data = VolumeData {
            electric: simulation.electric_buffer(),
            magnetic: simulation.magnetic_buffer(),
            peak: self.peak.into(),
            view: self.params(extent, settings, size, march),
        };

        {
            let mut pass = encoder.render(
                "volume",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: target,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            );
            let mut pipeline = pass.with(&self.volume);
            pipeline.bind(0, &data);
            pipeline.draw(0, 3, 0, 1);
        }
    }

    /// The auto-ranged peak energy density currently in use.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn destroy(&mut self) {
        self.context.destroy_buffer(self.peak);
        self.context.destroy_render_pipeline(&mut self.volume);
        self.context.destroy_compute_pipeline(&mut self.measure);
    }
}

/// Path length, in cells, over which a feature at the top of the range
/// accumulates to full brightness.
///
/// Emission along a ray is an integral, so the natural unit is a length. A
/// third of the domain means a packet occupying a third of the box reads as
/// saturated and anything smaller reads as proportionally dimmer, which is
/// what makes a wave packet look like an object rather than a fog.
fn reference_path(extent: Extent) -> f32 {
    let longest = extent.x.max(extent.y).max(extent.z) as f32;
    (longest / 3.0).max(1.0)
}

/// An offscreen colour target that can be written out as a PNG.
///
/// This is how the renderer is exercised in CI, and how the visual output can
/// be checked on a machine with no display at all.
pub struct Capture {
    context: Arc<gpu::Context>,
    texture: gpu::Texture,
    view: gpu::TextureView,
    readback: gpu::Buffer,
    size: gpu::Extent,
    initialized: bool,
}

impl Capture {
    /// sRGB, so the hardware does the encoding and the bytes that reach the
    /// PNG are already in the space a viewer expects.
    pub const FORMAT: gpu::TextureFormat = gpu::TextureFormat::Rgba8UnormSrgb;

    pub fn new(context: Arc<gpu::Context>, width: u32, height: u32) -> Self {
        let size = gpu::Extent {
            width,
            height,
            depth: 1,
        };
        let texture = context.create_texture(gpu::TextureDesc {
            name: "capture",
            format: Self::FORMAT,
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "capture",
                format: Self::FORMAT,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "capture readback",
            size: u64::from(width) * u64::from(height) * 4,
            memory: gpu::Memory::Shared,
        });
        Self {
            context,
            texture,
            view,
            readback,
            size,
            initialized: false,
        }
    }

    pub fn view(&self) -> gpu::TextureView {
        self.view
    }

    pub fn size(&self) -> gpu::Extent {
        self.size
    }

    /// Must run once before the texture is first used as a target.
    pub fn initialize(&mut self, encoder: &mut gpu::CommandEncoder) {
        if !self.initialized {
            encoder.init_texture(self.texture);
            self.initialized = true;
        }
    }

    pub fn copy_out(&self, encoder: &mut gpu::CommandEncoder) {
        let mut pass = encoder.transfer("capture readback");
        pass.copy_texture_to_buffer(
            self.texture.into(),
            self.readback.into(),
            self.size.width * 4,
            self.size,
        );
    }

    /// Writes the last captured frame. Call only after the submission that
    /// ran [`Self::copy_out`] has completed.
    pub fn write_png(&self, path: &Path) -> io::Result<()> {
        self.context.sync_buffer(self.readback);
        let byte_count = (self.size.width * self.size.height * 4) as usize;
        // SAFETY: `readback` is host-visible and holds exactly this many bytes,
        // written by a copy the caller has waited on.
        let pixels = unsafe { std::slice::from_raw_parts(self.readback.data(), byte_count) };

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = io::BufWriter::new(fs::File::create(path)?);
        let mut encoder = png::Encoder::new(file, self.size.width, self.size.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        let mut writer = encoder
            .write_header()
            .map_err(|error| io::Error::other(error.to_string()))?;
        writer
            .write_image_data(pixels)
            .map_err(|error| io::Error::other(error.to_string()))?;
        Ok(())
    }

    pub fn destroy(&mut self) {
        self.context.destroy_buffer(self.readback);
        self.context.destroy_texture_view(self.view);
        self.context.destroy_texture(self.texture);
    }
}
