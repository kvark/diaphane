//! The solver that has to be fast: WGSL compute kernels driven through
//! `blade-graphics`.
//!
//! # Shape of a step
//!
//! ```text
//! dispatch update_magnetic  →  barrier
//! dispatch update_electric  →  barrier
//! dispatch inject (once per source)
//! ```
//!
//! Three dispatches, no ping-pong. The `H` update reads only `E` and the `E`
//! update reads only `H`, so both write in place; double buffering would double
//! the memory traffic and buy nothing. FDTD is a bandwidth-bound stencil, and the
//! only optimization that matters is not moving bytes twice.
//!
//! # Many steps per submit
//!
//! [`Simulation::advance_by`] encodes every requested step into a *single* command
//! buffer. At interactive rates the solver runs hundreds of steps per displayed
//! frame, and a submit-and-wait per step would spend all of its time on
//! round-trips rather than on arithmetic. Nothing is read back between steps; the
//! renderer binds the field buffers directly.
//!
//! # Storage layout
//!
//! Fields are storage buffers of `f32`, not storage textures — the fragment shader
//! can then read them with no format negotiation and no copy. Components are
//! laid out one after another, so component `a` occupies
//! `[a·cell_count, (a+1)·cell_count)`.

use crate::{
    boundary::AbsorbingProfile,
    grid::{Axis, Grid},
    scene::Scene,
    source::{Injection, Source},
    timeline::{Snapshot, Steppable},
};
use blade_graphics as gpu;
use std::{mem, ptr, sync::Arc};

/// Uniform block shared by all three kernels.
///
/// Laid out as `vec4`s so the WGSL and Rust views agree without any padding
/// arithmetic; `Shader::check_struct_size` asserts they do.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    /// `nx, ny, nz, cell_count`.
    extent: [u32; 4],
    /// Source origin `xyz`, then the driven component.
    source_region: [u32; 4],
    /// Source extent `xyz`, then padding.
    source_extent: [u32; 4],
    /// Apodization centre `xyz`, then `1/waist²`.
    source_shape: [f32; 4],
    /// `amplitude · waveform(t)`, then the direction of time: `+1.0` steps
    /// forward, `-1.0` undoes a step. See the WGSL side for why one number
    /// is the entire reversal feature.
    source_drive: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct IntensityData {
    electric: gpu::BufferPiece,
    intensity: gpu::BufferPiece,
    params: Params,
}

#[derive(blade_macros::ShaderData)]
struct FieldData {
    electric: gpu::BufferPiece,
    magnetic: gpu::BufferPiece,
    material_index: gpu::BufferPiece,
    coefficients: gpu::BufferPiece,
    absorber: gpu::BufferPiece,
    geometry: gpu::BufferPiece,
    params: Params,
}

struct Pipelines {
    magnetic: gpu::ComputePipeline,
    electric: gpu::ComputePipeline,
    inject: gpu::ComputePipeline,
    intensity: gpu::ComputePipeline,
}

/// Encodes one injection into an open compute pass: uniforms, a dispatch over
/// the affected region, and the barrier ordering it against whatever reads
/// the field next. The forward and reverse paths share it; `direction` is the
/// sign of time, and the drive flips with it so retraction is exact. The
/// caller's `params.source_drive.y` — which the update kernels read as the
/// direction — is left equal to `direction`.
fn encode_injection(
    pass: &mut gpu::ComputeCommandEncoder<'_>,
    pipelines: &Pipelines,
    buffers: &Buffers,
    params: &mut Params,
    injection: &Injection,
    direction: f32,
) {
    params.source_region = [
        injection.origin[0] as u32,
        injection.origin[1] as u32,
        injection.origin[2] as u32,
        injection.component as u32,
    ];
    params.source_extent = [
        injection.extent[0] as u32,
        injection.extent[1] as u32,
        injection.extent[2] as u32,
        injection.magnetic as u32,
    ];
    params.source_shape = [
        injection.center[0],
        injection.center[1],
        injection.center[2],
        injection.inverse_waist_squared,
    ];
    params.source_drive = [direction * injection.value, direction, 0.0, 0.0];

    let region = gpu::Extent {
        width: injection.extent[0] as u32,
        height: injection.extent[1] as u32,
        depth: injection.extent[2] as u32,
    };
    // The pipeline context is scoped so it is dropped before the barrier. On
    // Metal it carries a `Drop` that ends the encoding, so holding it across
    // another `pass` call is a borrow error there — while on Vulkan the same
    // code compiles happily. Backend-specific borrow behaviour is invisible
    // until the other platform builds it.
    {
        let mut encoder = pass.with(&pipelines.inject);
        encoder.bind(0, &FieldData::new(buffers, *params));
        encoder.dispatch(pipelines.inject.get_dispatch_for(region));
    }
    // Sources may overlap, and each dispatch reads what the previous wrote.
    pass.barrier();
}

struct Buffers {
    electric: gpu::Buffer,
    magnetic: gpu::Buffer,
    material_index: gpu::Buffer,
    coefficients: gpu::Buffer,
    absorber: gpu::Buffer,
    geometry: gpu::Buffer,
    /// World-to-cell lookup, for a renderer marching a graded grid.
    lookup: gpu::Buffer,
    /// Running Σ|E|² per cell, fed by `accumulate_intensity` while a viewer
    /// wants the time-averaged view. Allocated always (one f32 per cell),
    /// dispatched only on demand.
    intensity: gpu::Buffer,
    /// Host-visible destination for [`Simulation::read_electric`] and friends.
    readback: gpu::Buffer,
}

/// A running simulation on the GPU.
pub struct Simulation {
    context: Arc<gpu::Context>,
    buffers: Buffers,
    pipelines: Pipelines,
    encoder: gpu::CommandEncoder,
    sync_point: Option<gpu::SyncPoint>,
    grid: Grid,
    sources: Vec<Source>,
    params: Params,
    field_bytes: u64,
    lookup_samples: u32,
    step: u64,
    /// Decided once from the scene, because [`Self::reverse_by`] must refuse
    /// a lossy scene for the same reason the CPU solver does.
    reversible: bool,
    /// Whether each step also accumulates |E|² into the intensity buffer.
    accumulating: bool,
    /// Steps summed into the intensity buffer so far; readers divide by it.
    accumulated: u64,
}

impl Simulation {
    /// How long [`Self::wait`] will block before giving up, in milliseconds.
    /// Generous, because a batch of a few thousand steps on a software
    /// rasterizer is not fast.
    const TIMEOUT_MS: u32 = 120_000;

    /// Builds the pipelines and uploads the scene.
    ///
    /// The context is shared rather than owned so that a renderer can bind the
    /// field buffers of a live simulation without a copy.
    pub fn new(context: Arc<gpu::Context>, scene: &Scene) -> Self {
        scene.grid.validate();
        let extent = scene.grid.extent;
        let cell_count = extent.total();
        let field_bytes = (3 * cell_count * mem::size_of::<f32>()) as u64;

        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("shaders/fdtd.wgsl"),
            naga_module: None,
        });
        shader.check_struct_size::<Params>();
        let layout = <FieldData as gpu::ShaderData>::layout();
        let pipeline = |name: &str, entry_point: &str| {
            context.create_compute_pipeline(gpu::ComputePipelineDesc {
                name,
                data_layouts: &[&layout],
                compute: shader.at(entry_point),
            })
        };
        let intensity_layout = <IntensityData as gpu::ShaderData>::layout();
        let pipelines = Pipelines {
            magnetic: pipeline("update_magnetic", "update_magnetic"),
            electric: pipeline("update_electric", "update_electric"),
            inject: pipeline("inject", "inject"),
            intensity: context.create_compute_pipeline(gpu::ComputePipelineDesc {
                name: "accumulate_intensity",
                data_layouts: &[&intensity_layout],
                compute: shader.at("accumulate_intensity"),
            }),
        };

        let indices = scene.material_indices();
        let coefficients = scene.materials.coefficients(&scene.grid);
        let absorption = AbsorbingProfile::new(&scene.grid, scene.boundary).packed();
        let geometry = scene.grid.packed_geometry();
        // Four samples per coarse cell, which puts a lookup interval well
        // inside one cell everywhere.
        let lookup_samples = (4 * extent.x.max(extent.y).max(extent.z)).clamp(256, 8192);
        let lookup = scene.grid.cell_lookup(lookup_samples);

        let device_buffer = |name: &str, size: u64| {
            context.create_buffer(gpu::BufferDesc {
                name,
                size,
                memory: gpu::Memory::Device,
            })
        };
        let buffers = Buffers {
            electric: device_buffer("electric", field_bytes),
            magnetic: device_buffer("magnetic", field_bytes),
            material_index: device_buffer("material index", byte_size(&indices)),
            coefficients: device_buffer("material coefficients", byte_size(&coefficients)),
            absorber: device_buffer("absorber profile", byte_size(&absorption)),
            geometry: device_buffer("axis geometry", byte_size(&geometry)),
            lookup: device_buffer("world to cell", byte_size(&lookup)),
            intensity: device_buffer("intensity", (cell_count * mem::size_of::<f32>()) as u64),
            readback: context.create_buffer(gpu::BufferDesc {
                name: "readback",
                size: field_bytes,
                memory: gpu::Memory::Shared,
            }),
        };

        let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "diaphane",
            // One buffer in flight being encoded while another runs.
            buffer_count: 2,
        });

        // Everything constant for the run goes up in one staging buffer and
        // one transfer pass.
        let mut staging_data = Vec::new();
        let material_at = append(&mut staging_data, &indices);
        let coefficients_at = append(&mut staging_data, &coefficients);
        let absorber_at = append(&mut staging_data, &absorption);
        let geometry_at = append(&mut staging_data, &geometry);
        let lookup_at = append(&mut staging_data, &lookup);
        let staging = context.create_buffer(gpu::BufferDesc {
            name: "scene upload",
            size: staging_data.len() as u64,
            memory: gpu::Memory::Upload,
        });
        // SAFETY: `staging` is host-visible upload memory, freshly created and
        // not yet referenced by any submitted command, and the destination
        // was allocated with exactly `staging_data.len()` bytes.
        unsafe {
            ptr::copy_nonoverlapping(staging_data.as_ptr(), staging.data(), staging_data.len());
        }

        encoder.start();
        {
            let mut pass = encoder.transfer("upload scene");
            pass.fill_buffer(buffers.electric.into(), field_bytes, 0);
            pass.fill_buffer(buffers.magnetic.into(), field_bytes, 0);
            pass.fill_buffer(
                buffers.intensity.into(),
                (cell_count * mem::size_of::<f32>()) as u64,
                0,
            );
            pass.copy_buffer_to_buffer(
                staging.at(material_at),
                buffers.material_index.into(),
                byte_size(&indices),
            );
            pass.copy_buffer_to_buffer(
                staging.at(coefficients_at),
                buffers.coefficients.into(),
                byte_size(&coefficients),
            );
            pass.copy_buffer_to_buffer(
                staging.at(absorber_at),
                buffers.absorber.into(),
                byte_size(&absorption),
            );
            pass.copy_buffer_to_buffer(
                staging.at(geometry_at),
                buffers.geometry.into(),
                byte_size(&geometry),
            );
            pass.copy_buffer_to_buffer(
                staging.at(lookup_at),
                buffers.lookup.into(),
                byte_size(&lookup),
            );
        }
        let sync_point = context.submit(&mut encoder);
        context
            .wait_for(&sync_point, Self::TIMEOUT_MS)
            .expect("scene upload failed");
        context.destroy_buffer(staging);

        Self {
            params: Params {
                extent: [extent.x, extent.y, extent.z, cell_count as u32],
                ..Default::default()
            },
            context,
            buffers,
            pipelines,
            encoder,
            sync_point: Some(sync_point),
            grid: scene.grid.clone(),
            sources: scene.sources.clone(),
            field_bytes,
            lookup_samples,
            step: 0,
            reversible: scene.is_reversible(),
            accumulating: false,
            accumulated: 0,
        }
    }

    /// World-to-cell lookup for the renderer, and how many samples per axis
    /// it holds. Exposed the same way the field buffers are: the renderer
    /// binds the simulation's own allocations rather than copying them.
    pub fn lookup_buffer(&self) -> gpu::BufferPiece {
        self.buffers.lookup.into()
    }

    pub fn lookup_samples(&self) -> u32 {
        self.lookup_samples
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Simulated time in seconds, matching [`crate::cpu::Simulation::time`].
    pub fn time(&self) -> f32 {
        self.step as f32 * self.grid.time_step()
    }

    /// The electric field buffer, for a renderer to bind directly.
    pub fn electric_buffer(&self) -> gpu::BufferPiece {
        self.buffers.electric.into()
    }

    /// The magnetic field buffer.
    pub fn magnetic_buffer(&self) -> gpu::BufferPiece {
        self.buffers.magnetic.into()
    }

    /// Bytes occupied by one field, all three components together.
    pub fn field_bytes(&self) -> u64 {
        self.field_bytes
    }

    /// Bytes of field traffic one full step must move, at best: every
    /// component of both fields is read once and written once.
    ///
    /// Used to turn a step rate into an effective bandwidth, which is the
    /// number that says whether the solver is close to the hardware limit.
    pub fn bytes_per_step(&self) -> u64 {
        4 * self.field_bytes
    }

    /// Encodes and submits `steps` complete time steps as one command buffer.
    ///
    /// Returns immediately; call [`Self::wait`] before reading anything back.
    pub fn advance_by(&mut self, steps: u64) {
        if steps == 0 {
            return;
        }
        // Reuse of a command buffer requires the previous submission to have
        // retired.
        self.wait();

        let Self {
            ref context,
            ref buffers,
            ref pipelines,
            ref mut encoder,
            ref grid,
            ref sources,
            ref mut params,
            ref mut step,
            ref accumulating,
            ref mut accumulated,
            ..
        } = *self;

        let extent = gpu::Extent {
            width: grid.extent.x,
            height: grid.extent.y,
            depth: grid.extent.z,
        };
        let time_step = grid.time_step();

        encoder.start();
        {
            let mut pass = encoder.compute("advance");
            for _ in 0..steps {
                // `y` is the direction of time; the update kernels read it,
                // and every encoded injection leaves it at `1.0`.
                params.source_drive = [0.0, 1.0, 0.0, 0.0];
                // Where `E` will stand after its update; injections state
                // their own instants relative to it.
                let time = (*step + 1) as f32 * time_step;
                {
                    let mut encoder = pass.with(&pipelines.magnetic);
                    encoder.bind(0, &FieldData::new(buffers, *params));
                    encoder.dispatch(pipelines.magnetic.get_dispatch_for(extent));
                }
                pass.barrier();
                // Magnetic injections land *between* the half-updates: a
                // plane wave's scattered-side correction has to be in place
                // before the electric update reads the row it repairs, or the
                // rows beside it ingest incident field they must never see.
                for source in sources.iter() {
                    for injection in source.injections(grid, time) {
                        if injection.value != 0.0 && injection.magnetic {
                            encode_injection(
                                &mut pass, pipelines, buffers, params, &injection, 1.0,
                            );
                        }
                    }
                }
                {
                    let mut encoder = pass.with(&pipelines.electric);
                    encoder.bind(0, &FieldData::new(buffers, *params));
                    encoder.dispatch(pipelines.electric.get_dispatch_for(extent));
                }
                pass.barrier();
                for source in sources.iter() {
                    for injection in source.injections(grid, time) {
                        if injection.value != 0.0 && !injection.magnetic {
                            encode_injection(
                                &mut pass, pipelines, buffers, params, &injection, 1.0,
                            );
                        }
                    }
                }
                if *accumulating {
                    // `E` is complete for this step; add its square in.
                    let mut encoder = pass.with(&pipelines.intensity);
                    encoder.bind(
                        0,
                        &IntensityData {
                            electric: buffers.electric.into(),
                            intensity: buffers.intensity.into(),
                            params: *params,
                        },
                    );
                    encoder.dispatch(pipelines.intensity.get_dispatch_for(extent));
                    *accumulated += 1;
                }
                *step += 1;
            }
        }
        self.sync_point = Some(context.submit(encoder));
    }

    /// Undoes `steps` complete time steps, encoded as one command buffer.
    ///
    /// The same involution the CPU solver performs, run by the same kernels:
    /// with nothing lossy in the scene, the inverse update is the forward
    /// update with the gain negated and the two half-steps in the opposite
    /// order, so the `direction` slot of [`Params`] is the entire feature.
    /// Sources come back out exactly, because a waveform is a pure function
    /// of time.
    ///
    /// Panics on a lossy scene, exactly like [`crate::cpu::Simulation::reverse`]
    /// and for the same reason: a dissipative update run backwards amplifies
    /// roundoff exponentially, and returning that would look like a solver
    /// bug rather than a refusal.
    pub fn reverse_by(&mut self, steps: u64) {
        if steps == 0 {
            return;
        }
        assert!(
            self.reversible,
            "this scene is lossy, and running a dissipative update backwards \
             amplifies roundoff exponentially; reversal needs Boundary::Pec \
             and no conductive material"
        );
        assert!(self.step >= steps, "cannot step back past the start");
        // A sum over steps that are about to be un-stepped is not an average
        // of anything; reversal restarts the accumulation.
        self.clear_intensity();
        self.wait();

        let Self {
            ref context,
            ref buffers,
            ref pipelines,
            ref mut encoder,
            ref grid,
            ref sources,
            ref mut params,
            ref mut step,
            ..
        } = *self;

        let extent = gpu::Extent {
            width: grid.extent.x,
            height: grid.extent.y,
            depth: grid.extent.z,
        };
        let time_step = grid.time_step();

        encoder.start();
        {
            let mut pass = encoder.compute("reverse");
            for _ in 0..steps {
                // Forward order is H, inject-H, E, inject-E; backward undoes
                // each in the opposite order, amplitudes negated.
                params.source_drive = [0.0, -1.0, 0.0, 0.0];
                let time = *step as f32 * time_step;
                for source in sources.iter() {
                    for injection in source.injections(grid, time) {
                        if injection.value != 0.0 && !injection.magnetic {
                            encode_injection(
                                &mut pass, pipelines, buffers, params, &injection, -1.0,
                            );
                        }
                    }
                }
                {
                    let mut encoder = pass.with(&pipelines.electric);
                    encoder.bind(0, &FieldData::new(buffers, *params));
                    encoder.dispatch(pipelines.electric.get_dispatch_for(extent));
                }
                pass.barrier();
                for source in sources.iter() {
                    for injection in source.injections(grid, time) {
                        if injection.value != 0.0 && injection.magnetic {
                            encode_injection(
                                &mut pass, pipelines, buffers, params, &injection, -1.0,
                            );
                        }
                    }
                }
                {
                    let mut encoder = pass.with(&pipelines.magnetic);
                    encoder.bind(0, &FieldData::new(buffers, *params));
                    encoder.dispatch(pipelines.magnetic.get_dispatch_for(extent));
                }
                pass.barrier();
                *step -= 1;
            }
        }
        self.sync_point = Some(context.submit(encoder));
    }

    /// Turns per-step intensity accumulation on or off.
    ///
    /// Turning it on starts a fresh average -- the buffer is zeroed -- and
    /// each subsequent step adds `|E|²` per cell. Off costs nothing: the
    /// dispatch simply never happens. Readers normalize by
    /// [`Self::accumulated_steps`].
    pub fn set_accumulate_intensity(&mut self, on: bool) {
        if on && !self.accumulating {
            self.clear_intensity();
        }
        self.accumulating = on;
    }

    /// Steps summed into the intensity buffer since it was last cleared.
    pub fn accumulated_steps(&self) -> u64 {
        self.accumulated
    }

    /// The running `Σ|E|²` buffer, one `f32` per cell, for a renderer.
    pub fn intensity_buffer(&self) -> gpu::BufferPiece {
        self.buffers.intensity.into()
    }

    fn clear_intensity(&mut self) {
        if self.accumulated == 0 {
            return;
        }
        self.wait();
        self.encoder.start();
        {
            let mut pass = self.encoder.transfer("clear intensity");
            pass.fill_buffer(self.buffers.intensity.into(), self.field_bytes / 3, 0);
        }
        self.sync_point = Some(self.context.submit(&mut self.encoder));
        self.accumulated = 0;
    }

    /// Blocks until the last submission has retired.
    pub fn wait(&mut self) {
        if let Some(sync_point) = self.sync_point.take() {
            self.context
                .wait_for(&sync_point, Self::TIMEOUT_MS)
                .expect("the GPU did not finish in time");
        }
    }

    /// Clears the fields and rewinds the clock, leaving geometry alone.
    pub fn reset(&mut self) {
        self.wait();
        self.encoder.start();
        {
            let mut pass = self.encoder.transfer("reset fields");
            pass.fill_buffer(self.buffers.electric.into(), self.field_bytes, 0);
            pass.fill_buffer(self.buffers.magnetic.into(), self.field_bytes, 0);
            pass.fill_buffer(self.buffers.intensity.into(), self.field_bytes / 3, 0);
        }
        self.sync_point = Some(self.context.submit(&mut self.encoder));
        self.step = 0;
        self.accumulated = 0;
    }

    /// Copies the electric field back to the host: `3 · cell_count` values,
    /// component-major, so `axis` occupies `[a·cells, (a+1)·cells)`.
    ///
    /// Synchronous and slow by design — this is for tests and diagnostics, not
    /// for the render loop, which binds the buffers directly.
    pub fn read_electric(&mut self) -> Vec<f32> {
        self.read(self.buffers.electric)
    }

    pub fn read_magnetic(&mut self) -> Vec<f32> {
        self.read(self.buffers.magnetic)
    }

    /// One component out of a slice returned by [`Self::read_electric`].
    pub fn component<'a>(&self, values: &'a [f32], axis: Axis) -> &'a [f32] {
        let cells = self.grid.extent.total();
        &values[axis.index() * cells..(axis.index() + 1) * cells]
    }

    /// Uploads a field back to the device, for [`Steppable::restore`].
    ///
    /// Goes through the host-visible readback buffer in the other direction
    /// rather than allocating a staging buffer per call — a restore happens on
    /// a seek, not in the render loop.
    fn write(&mut self, destination: gpu::Buffer, values: &[f32]) {
        self.wait();
        assert_eq!(
            values.len(),
            3 * self.grid.extent.total(),
            "snapshot was taken from a different grid"
        );
        // SAFETY: `readback` is host-visible, holds exactly `field_bytes`, and
        // the wait above means no submitted command is still reading it.
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.buffers.readback.data().cast::<f32>(),
                values.len(),
            );
        }
        self.context.sync_buffer(self.buffers.readback);

        self.encoder.start();
        {
            let mut pass = self.encoder.transfer("restore");
            pass.copy_buffer_to_buffer(
                self.buffers.readback.into(),
                destination.into(),
                self.field_bytes,
            );
        }
        let sync_point = self.context.submit(&mut self.encoder);
        self.context
            .wait_for(&sync_point, Self::TIMEOUT_MS)
            .expect("restore failed");
        self.sync_point = Some(sync_point);
    }

    fn read(&mut self, source: gpu::Buffer) -> Vec<f32> {
        self.read_bytes(source, self.field_bytes)
    }

    /// The accumulated `Σ|E|²` per cell — see [`Self::set_accumulate_intensity`].
    pub fn read_intensity(&mut self) -> Vec<f32> {
        let bytes = self.field_bytes / 3;
        self.read_bytes(self.buffers.intensity, bytes)
    }

    fn read_bytes(&mut self, source: gpu::Buffer, bytes: u64) -> Vec<f32> {
        self.wait();
        self.encoder.start();
        {
            let mut pass = self.encoder.transfer("readback");
            pass.copy_buffer_to_buffer(source.into(), self.buffers.readback.into(), bytes);
        }
        let sync_point = self.context.submit(&mut self.encoder);
        self.context
            .wait_for(&sync_point, Self::TIMEOUT_MS)
            .expect("readback failed");
        self.context.sync_buffer(self.buffers.readback);
        self.sync_point = Some(sync_point);

        let count = bytes as usize / std::mem::size_of::<f32>();
        // SAFETY: `readback` is host-visible, holds at least `count` `f32`s
        // that the copy above has finished writing, and is not aliased
        // elsewhere.
        let mapped = unsafe {
            std::slice::from_raw_parts(self.buffers.readback.data().cast::<f32>(), count)
        };
        mapped.to_vec()
    }
}

/// Snapshots are a full readback and restores a full upload, which is why a
/// [`crate::timeline::Timeline`] takes them on keyframe boundaries rather than
/// every step.
impl Steppable for Simulation {
    fn step_count(&self) -> u64 {
        self.step
    }

    fn advance_by(&mut self, steps: u64) {
        Self::advance_by(self, steps);
    }

    fn reset(&mut self) {
        Self::reset(self);
    }

    fn snapshot(&mut self) -> Snapshot {
        Snapshot {
            step: self.step,
            electric: self.read_electric(),
            magnetic: self.read_magnetic(),
        }
    }

    fn restore(&mut self, snapshot: &Snapshot) {
        // The average belongs to the history that was just abandoned.
        self.clear_intensity();
        self.write(self.buffers.electric, &snapshot.electric);
        self.write(self.buffers.magnetic, &snapshot.magnetic);
        self.step = snapshot.step;
    }
}

impl FieldData {
    fn new(buffers: &Buffers, params: Params) -> Self {
        Self {
            electric: buffers.electric.into(),
            magnetic: buffers.magnetic.into(),
            material_index: buffers.material_index.into(),
            coefficients: buffers.coefficients.into(),
            absorber: buffers.absorber.into(),
            geometry: buffers.geometry.into(),
            params,
        }
    }
}

impl Drop for Simulation {
    fn drop(&mut self) {
        self.wait();
        self.context.destroy_buffer(self.buffers.electric);
        self.context.destroy_buffer(self.buffers.magnetic);
        self.context.destroy_buffer(self.buffers.material_index);
        self.context.destroy_buffer(self.buffers.coefficients);
        self.context.destroy_buffer(self.buffers.absorber);
        self.context.destroy_buffer(self.buffers.geometry);
        self.context.destroy_buffer(self.buffers.lookup);
        self.context.destroy_buffer(self.buffers.intensity);
        self.context.destroy_buffer(self.buffers.readback);
        self.context
            .destroy_compute_pipeline(&mut self.pipelines.magnetic);
        self.context
            .destroy_compute_pipeline(&mut self.pipelines.electric);
        self.context
            .destroy_compute_pipeline(&mut self.pipelines.inject);
        self.context.destroy_command_encoder(&mut self.encoder);
    }
}

/// Opens a headless context, or explains why it could not.
///
/// Separated out so tests and benchmarks can skip cleanly on a machine with no
/// usable device rather than failing.
pub fn headless_context() -> Result<Arc<gpu::Context>, gpu::NotSupportedError> {
    // SAFETY: `Context::init` is unsafe because it loads the platform driver;
    // there is no additional precondition for the caller beyond calling it
    // once per context.
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            presentation: false,
            validation: cfg!(debug_assertions),
            ..Default::default()
        })
    }?;
    Ok(Arc::new(context))
}

fn byte_size<T>(values: &[T]) -> u64 {
    mem::size_of_val(values) as u64
}

/// Appends `values` to a staging blob, padded so the next section starts on a
/// storage-buffer-aligned offset, and returns the offset it was written at.
fn append<T: bytemuck::Pod>(blob: &mut Vec<u8>, values: &[T]) -> u64 {
    let alignment = gpu::limits::STORAGE_BUFFER_ALIGNMENT as usize;
    let offset = blob.len();
    blob.extend_from_slice(bytemuck::cast_slice(values));
    let padding = blob.len().next_multiple_of(alignment) - blob.len();
    blob.resize(blob.len() + padding, 0);
    offset as u64
}
