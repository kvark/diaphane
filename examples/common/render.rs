//! Volume rendering of a live simulation.
//!
//! The fragment shader binds the solver's own field buffers. Nothing is
//! copied, converted, or staged between the compute kernel writing a field and
//! the pixel that displays it.

use crate::common::camera::Camera;
use blade_graphics as gpu;
use diaphane::gpu::Simulation;
use std::{fs, io, mem, path::Path, sync::Arc};

/// Which quantity the volume shows.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ViewMode {
    /// Signed `E` and `H` at once, in two hues, each along the component that
    /// carries it.
    ///
    /// The default, because it is the only view in which the two fields are
    /// separately visible. [`Self::Energy`] tints their *densities*, and in a
    /// travelling wave those are equal — so the tints sum to white and the
    /// packet reads as a colourless blob. The energy split is the right view
    /// for a cavity, where the two genuinely alternate.
    #[default]
    Fields,
    /// Electric and magnetic energy density in opposing hues, which is where
    /// a standing wave visibly trades energy between them.
    Energy,
    /// The dominant `E` component, signed, through a diverging colormap.
    /// Which component that is gets measured per frame, not assumed.
    Electric,
    /// The dominant `H` component, signed.
    Magnetic,
    /// Total energy density, monochrome.
    Magnitude,
    /// The mesh, with no field in it. Shows where the resolution went.
    Grid,
    /// The textbook figure: `E` and `H` plotted as ribbons in perpendicular
    /// planes along the direction of travel.
    ///
    /// The only view that shows the two fields at right angles, and it has to
    /// be a graph to do it — see the note on `MODE_RIBBONS` in the shader.
    Ribbons,
    /// Time-averaged `|E|²` — what a detector integrates. The solver
    /// accumulates it only while this view is up, so every other mode costs
    /// nothing; switching to it starts a fresh average.
    Intensity,
}

impl ViewMode {
    pub const ALL: [Self; 8] = [
        Self::Fields,
        Self::Energy,
        Self::Electric,
        Self::Magnetic,
        Self::Magnitude,
        Self::Grid,
        Self::Ribbons,
        Self::Intensity,
    ];

    pub fn label(self) -> &'static str {
        // "signed E", not "Ez": the renderer displays whichever component
        // carries the field, measured per frame, so naming one would lie in
        // any scene that is not z-polarized.
        match self {
            Self::Fields => "E and H",
            Self::Energy => "energy split",
            Self::Electric => "signed E",
            Self::Magnetic => "signed H",
            Self::Magnitude => "total energy",
            Self::Grid => "the grid",
            Self::Ribbons => "E and H as ribbons",
            Self::Intensity => "time-averaged intensity",
        }
    }

    fn code(self) -> f32 {
        match self {
            Self::Fields => 4.0,
            Self::Energy => 0.0,
            Self::Electric => 1.0,
            Self::Magnetic => 2.0,
            Self::Magnitude => 3.0,
            Self::Grid => 5.0,
            Self::Ribbons => 6.0,
            Self::Intensity => 7.0,
        }
    }

    fn is_signed(self) -> bool {
        match self {
            // Ribbons deflect by a field *amplitude*, so they need the
            // amplitude exposure like every other signed view -- the energy
            // scale would leave a spurious factor of the field peak in the
            // deflection, blowing the ribbons into planes in quiet scenes and
            // collapsing them in loud ones.
            Self::Electric | Self::Magnetic | Self::Ribbons => true,
            Self::Energy | Self::Magnitude | Self::Grid | Self::Intensity => false,
            Self::Fields => true,
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
    /// What [`Self::toggle_log`] restores: the last nonzero strength, so a
    /// `--log` chosen on the command line survives an off/on round trip
    /// instead of being replaced by a hardcoded figure.
    remembered_log: f32,
    /// The scrub bar, or `None` to hide it.
    pub scrub: Option<ScrubBar>,
}

/// What the scrub bar should show, as fractions of the run so far.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScrubBar {
    /// Where the playhead sits, 0 to 1.
    pub played: f32,
    /// Where the keyframed window begins — the earliest step that can be
    /// reached without replaying from zero.
    pub window_start: f32,
}

impl ScrubBar {
    /// Height of the bar as a fraction of the image. The shader draws with
    /// the same constant. This bar is display only — the window's interactive
    /// scrubbing is the egui slider, so there is no hit test to keep in sync.
    pub const HEIGHT: f32 = 0.045;
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
            remembered_log: 6.0,
            scrub: None,
        }
    }

    /// Turns the signed-log compression off and back on to the strength it
    /// had, whether that came from the default or from `--log`.
    pub fn toggle_log(&mut self) {
        if self.log_strength > 0.0 {
            self.remembered_log = self.log_strength;
            self.log_strength = 0.0;
        } else {
            self.log_strength = self.remembered_log;
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
    /// Domain size in coarse cells, then the lookup sample count per axis.
    box_size: [f32; 4],
    /// Camera position in cells, then the ray-march step in cells.
    origin: [f32; 4],
    /// Camera right scaled by aspect and field of view, then exposure.
    right: [f32; 4],
    /// Camera up scaled by field of view, then the signed-log strength.
    up: [f32; 4],
    /// Camera forward, then the view mode.
    forward: [f32; 4],
    /// Reciprocal of the reference path length, then the scrub bar: played
    /// fraction, window start fraction, and bar height (0 hides it).
    tone: [f32; 4],
    /// Which component of `E` and of `H` the signed views read, then padding.
    components: [u32; 4],
}

#[derive(blade_macros::ShaderData)]
struct VolumeData {
    electric: gpu::BufferPiece,
    magnetic: gpu::BufferPiece,
    peak: gpu::BufferPiece,
    lookup: gpu::BufferPiece,
    intensity: gpu::BufferPiece,
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
    /// Component of `E` and of `H` the signed views read.
    ///
    /// Measured rather than assumed. A wave along x polarized in z puts its
    /// magnetic field in y, so a fixed component pairs one real field with one
    /// that is a hundredth of it — and which components those are is a fact
    /// about the scene, not about the renderer.
    components: [u32; 2],
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
            size: PEAK_SLOTS as u64 * mem::size_of::<u32>() as u64,
            memory: gpu::Memory::Shared,
        });

        Self {
            context,
            volume,
            measure,
            peak,
            scale: 0.0,
            components: [2, 1],
        }
    }

    /// Reads the previous frame's reduction and folds it into the running
    /// scale.
    ///
    /// Rises quickly and falls slowly. Without that asymmetry the display
    /// flickers as a pulse's amplitude swings: every dip in the peak would be
    /// answered by an immediate brightening.
    /// `advanced` is false when the solver did not step, and then the range is
    /// left alone.
    ///
    /// Otherwise a paused viewer keeps changing: the smoothing converges toward
    /// the measured peak over roughly a hundred frames, so the image goes on
    /// "developing" with the field frozen. That reads as the simulation still
    /// running, which is the one thing a pause has to rule out.
    fn update_scale(&mut self, advanced: bool, mode: ViewMode) {
        if !advanced {
            return;
        }
        // SAFETY: `peak` is host-visible, `PEAK_SLOTS` words long, and the
        // submission that wrote it has completed before this runs.
        let slots: [f32; PEAK_SLOTS] = std::array::from_fn(|slot| {
            f32::from_bits(unsafe { self.peak.data().cast::<u32>().add(slot).read_unaligned() })
        });
        // The averaged intensity ranges itself: its peak lives where the
        // fringes are brightest, which over a long average can sit far from
        // the instantaneous energy peak the other views expose against.
        let measured = if mode == ViewMode::Intensity {
            slots[7]
        } else {
            slots[0]
        };
        // Whichever component carries the most field is the one worth drawing.
        // Taken from the same reduction the exposure comes from, so it costs
        // nothing extra and tracks a scene whose polarization is not obvious.
        let dominant = |offset: usize| {
            (0..3)
                .filter(|axis| slots[offset + axis].is_finite())
                .max_by(|a, b| slots[offset + a].total_cmp(&slots[offset + b]))
                .unwrap_or(0) as u32
        };
        let electric = dominant(1);
        let mut magnetic = dominant(4);
        if magnetic == electric {
            // Degenerate only when a field is uniformly zero, but the ribbon
            // view derives the propagation axis as the one left over — and
            // `3 - a - a` is not an axis.
            magnetic = (electric + 1) % 3;
        }
        self.components = [electric, magnetic];
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
        simulation: &Simulation,
        settings: &ViewSettings,
        target: gpu::Extent,
        march: f32,
    ) -> ViewParams {
        let extent = simulation.grid().extent;
        let box_size = simulation.grid().box_size();
        let aspect = target.width as f32 / target.height.max(1) as f32;
        let basis = settings.camera.basis(box_size, aspect);
        // The reduction measures energy density; the signed views show a field
        // amplitude, whose scale is the square root of it.
        let reference = self.scale.max(f32::MIN_POSITIVE);
        let exposure = if settings.mode.is_signed() {
            settings.gain / (2.0 * reference).sqrt()
        } else {
            settings.gain / reference
        };
        let accumulated = simulation.accumulated_steps().min(u64::from(u32::MAX)) as u32;
        ViewParams {
            extent: [extent.x, extent.y, extent.z, extent.total() as u32],
            box_size: [
                box_size[0],
                box_size[1],
                box_size[2],
                simulation.lookup_samples() as f32,
            ],
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
            components: [self.components[0], self.components[1], accumulated, 0],
            tone: match settings.scrub {
                Some(scrub) => [
                    1.0 / reference_path(box_size),
                    scrub.played,
                    scrub.window_start,
                    ScrubBar::HEIGHT,
                ],
                None => [1.0 / reference_path(box_size), 0.0, 0.0, 0.0],
            },
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
        let accumulated = simulation.accumulated_steps().min(u64::from(u32::MAX)) as u32;
        let data = VolumeData {
            electric: simulation.electric_buffer(),
            magnetic: simulation.magnetic_buffer(),
            peak: self.peak.into(),
            lookup: simulation.lookup_buffer(),
            intensity: simulation.intensity_buffer(),
            view: ViewParams {
                extent: [extent.x, extent.y, extent.z, extent.total() as u32],
                components: [self.components[0], self.components[1], accumulated, 0],
                ..Default::default()
            },
        };
        {
            let mut pass = encoder.transfer("clear peak");
            pass.fill_buffer(
                self.peak.into(),
                PEAK_SLOTS as u64 * mem::size_of::<u32>() as u64,
                0,
            );
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
        advanced: bool,
    ) {
        self.update_scale(advanced, settings.mode);
        self.measure(encoder, simulation);

        // One sample per cell along the ray is enough at the resolutions the
        // solver is usable at; the field varies over tens of cells.
        let march = 1.0;
        let data = VolumeData {
            electric: simulation.electric_buffer(),
            magnetic: simulation.magnetic_buffer(),
            peak: self.peak.into(),
            lookup: simulation.lookup_buffer(),
            intensity: simulation.intensity_buffer(),
            view: self.params(simulation, settings, size, march),
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

/// Slots in the reduction buffer: the energy peak, then `|E|` and `|H|` per
/// component.
const PEAK_SLOTS: usize = 8;

/// Path length, in cells, over which a feature at the top of the range
/// accumulates to full brightness.
///
/// Emission along a ray is an integral, so the natural unit is a length. A
/// third of the domain means a packet occupying a third of the box reads as
/// saturated and anything smaller reads as proportionally dimmer, which is
/// what makes a wave packet look like an object rather than a fog.
fn reference_path(box_size: [f32; 3]) -> f32 {
    let longest = box_size[0].max(box_size[1]).max(box_size[2]);
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

    /// The last captured frame as RGBA bytes. Call only after the submission
    /// that ran [`Self::copy_out`] has completed.
    pub fn pixels(&self) -> &[u8] {
        self.context.sync_buffer(self.readback);
        let byte_count = (self.size.width * self.size.height * 4) as usize;
        // SAFETY: `readback` is host-visible and holds exactly this many bytes,
        // written by a copy the caller has waited on.
        unsafe { std::slice::from_raw_parts(self.readback.data(), byte_count) }
    }

    /// Writes the last captured frame.
    pub fn write_png(&self, path: &Path) -> io::Result<()> {
        let pixels = self.pixels();

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

/// Collects frames and writes them out as one animated GIF.
///
/// A GIF rather than a video because it plays inline in a README with no
/// player, no autoplay policy and no codec — the format is ancient and that is
/// exactly the point. The cost is 256 colours per frame, which a dark volume
/// render with two hues survives better than most content would.
pub struct Animation {
    frames: Vec<Vec<u8>>,
    size: (u16, u16),
    /// Hundredths of a second per frame, which is the only unit GIF has.
    delay: u16,
}

impl Animation {
    pub fn new(width: u32, height: u32, delay_centiseconds: u16) -> Self {
        // The format stores dimensions in sixteen bits; a silent truncation
        // here would corrupt every frame rather than refuse the first one.
        let side = |pixels: u32| u16::try_from(pixels).expect("GIF dimensions cap at 65535");
        Self {
            frames: Vec::new(),
            size: (side(width), side(height)),
            // 2 is the floor most viewers honour; below it they substitute
            // their own and the animation plays at an unrelated speed.
            delay: delay_centiseconds.max(2),
        }
    }

    pub fn push(&mut self, rgba: &[u8]) {
        self.frames.push(rgba.to_vec());
    }

    pub fn write(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = io::BufWriter::new(fs::File::create(path)?);
        let mut encoder = gif::Encoder::new(&mut file, self.size.0, self.size.1, &[])
            .map_err(|error| io::Error::other(error.to_string()))?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            .map_err(|error| io::Error::other(error.to_string()))?;
        for pixels in &self.frames {
            let mut copy = pixels.clone();
            // Quantizes per frame. A shared palette would be smaller and would
            // band less, but a wave packet's colours drift as it moves, so a
            // per-frame palette is what keeps the tail from posterizing.
            let mut frame = gif::Frame::from_rgba_speed(self.size.0, self.size.1, &mut copy, 10);
            frame.delay = self.delay;
            encoder
                .write_frame(&frame)
                .map_err(|error| io::Error::other(error.to_string()))?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}
