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
    source::Source,
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
    /// `amplitude · waveform(t)`, then padding.
    source_drive: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct FieldData {
    electric: gpu::BufferPiece,
    magnetic: gpu::BufferPiece,
    material_index: gpu::BufferPiece,
    coefficients: gpu::BufferPiece,
    absorber: gpu::BufferPiece,
    params: Params,
}

struct Pipelines {
    magnetic: gpu::ComputePipeline,
    electric: gpu::ComputePipeline,
    inject: gpu::ComputePipeline,
}

struct Buffers {
    electric: gpu::Buffer,
    magnetic: gpu::Buffer,
    material_index: gpu::Buffer,
    coefficients: gpu::Buffer,
    absorber: gpu::Buffer,
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
    step: u64,
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
        let pipelines = Pipelines {
            magnetic: pipeline("update_magnetic", "update_magnetic"),
            electric: pipeline("update_electric", "update_electric"),
            inject: pipeline("inject", "inject"),
        };

        let indices = scene.material_indices();
        let coefficients = scene.materials.coefficients(&scene.grid);
        let absorption = AbsorbingProfile::new(&scene.grid, scene.boundary).packed();

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
            grid: scene.grid,
            sources: scene.sources.clone(),
            field_bytes,
            step: 0,
        }
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
                params.source_drive = [0.0; 4];
                {
                    let mut encoder = pass.with(&pipelines.magnetic);
                    encoder.bind(0, &FieldData::new(buffers, *params));
                    encoder.dispatch(pipelines.magnetic.get_dispatch_for(extent));
                }
                pass.barrier();
                {
                    let mut encoder = pass.with(&pipelines.electric);
                    encoder.bind(0, &FieldData::new(buffers, *params));
                    encoder.dispatch(pipelines.electric.get_dispatch_for(extent));
                }
                pass.barrier();

                // `E` now holds `(step + 1)·Δt`, which is when the sources act.
                let time = (*step + 1) as f32 * time_step;
                for source in sources.iter() {
                    let injection = source.injection(grid, time);
                    if injection.value == 0.0 {
                        continue;
                    }
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
                        0,
                    ];
                    params.source_shape = [
                        injection.center[0],
                        injection.center[1],
                        injection.center[2],
                        injection.inverse_waist_squared,
                    ];
                    params.source_drive = [injection.value, 0.0, 0.0, 0.0];

                    let region = gpu::Extent {
                        width: injection.extent[0] as u32,
                        height: injection.extent[1] as u32,
                        depth: injection.extent[2] as u32,
                    };
                    let mut encoder = pass.with(&pipelines.inject);
                    encoder.bind(0, &FieldData::new(buffers, *params));
                    encoder.dispatch(pipelines.inject.get_dispatch_for(region));
                    // Sources may overlap, and each dispatch reads what the
                    // previous one wrote.
                    pass.barrier();
                }
                *step += 1;
            }
        }
        self.sync_point = Some(context.submit(encoder));
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
        }
        self.sync_point = Some(self.context.submit(&mut self.encoder));
        self.step = 0;
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

    fn read(&mut self, source: gpu::Buffer) -> Vec<f32> {
        self.wait();
        self.encoder.start();
        {
            let mut pass = self.encoder.transfer("readback");
            pass.copy_buffer_to_buffer(
                source.into(),
                self.buffers.readback.into(),
                self.field_bytes,
            );
        }
        let sync_point = self.context.submit(&mut self.encoder);
        self.context
            .wait_for(&sync_point, Self::TIMEOUT_MS)
            .expect("readback failed");
        self.context.sync_buffer(self.buffers.readback);
        self.sync_point = Some(sync_point);

        let count = 3 * self.grid.extent.total();
        // SAFETY: `readback` is host-visible, holds `count` `f32`s that the
        // copy above has finished writing, and is not aliased elsewhere.
        let mapped = unsafe {
            std::slice::from_raw_parts(self.buffers.readback.data().cast::<f32>(), count)
        };
        mapped.to_vec()
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
