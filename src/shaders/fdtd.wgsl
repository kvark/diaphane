// 3D Yee FDTD, impedance-normalized.
//
// The staggering convention is documented in `src/grid.rs` and is repeated here
// because this is where getting it wrong is fatal and invisible:
//
//   E[a] sits at a HALF position along axis `a`, integer along the other two.
//   H[a] sits at an INTEGER position along axis `a`, half along the other two.
//
// Both updates are written once and run for each axis `a`, with `b = a+1` and
// `c = a+2` cyclically:
//
//   H[a] -= (1/ur) * ( g_b*dE[c] - g_c*dE[b] )    forward differences
//   E[a] += (1/er) * ( g_b*dH[c] - g_c*dH[b] )    backward differences
//
// Forward for H and backward for E is what lands each derivative exactly on the
// sample it updates.
//
// `g` is `c*dt/delta` for the axis the difference is taken along, and it is a
// per-cell lookup because cells need not be the same size. H differences E
// between two corners -- one whole cell, the primary spacing -- while E
// differences H between two centres, which spans two half cells that may
// differ. On a uniform grid every entry is the Courant number and the two
// coincide, which is exactly why a graded grid is what finds a mistake here.
//
// Storage is component-major: component `a` of a field occupies
// `[a * cell_count, (a+1) * cell_count)`. Within a component the layout is
// x-major, so neighbouring threads along x touch neighbouring addresses.

struct Coefficients {
    electric_gain: f32,
    electric_loss: f32,
    magnetic_gain: f32,
    magnetic_loss: f32,
}

struct Params {
    // nx, ny, nz, cell_count
    extent: vec4<u32>,
    // source origin xyz, driven component
    source_region: vec4<u32>,
    // source extent xyz, unused
    source_extent: vec4<u32>,
    // apodization centre xyz, 1/waist^2
    source_shape: vec4<f32>,
    // amplitude * waveform(t), unused, unused, unused
    source_drive: vec4<f32>,
}

var<storage, read_write> electric: array<f32>;
var<storage, read_write> magnetic: array<f32>;
var<storage, read> material_index: array<u32>;
var<storage, read> coefficients: array<Coefficients>;
// Six 1D absorber profiles packed back to back: the integer-position profiles
// for x, y, z, then the half-position ones. Offsets follow from the extent, so
// nothing extra has to travel alongside.
var<storage, read> absorber: array<f32>;
// Per-axis geometry in three sections of three axes each: primary gains, dual
// gains, then cell centres in metres. Packed and documented in
// `Grid::packed_geometry`.
var<storage, read> geometry: array<f32>;
var<uniform> params: Params;

fn extent_of(axis: u32) -> u32 {
    if axis == 0u {
        return params.extent.x;
    } else if axis == 1u {
        return params.extent.y;
    }
    return params.extent.z;
}

fn stride_of(axis: u32) -> u32 {
    if axis == 0u {
        return 1u;
    } else if axis == 1u {
        return params.extent.x;
    }
    return params.extent.x * params.extent.y;
}

// Offset of one axis inside a section whose entries are one per cell.
fn axis_base(axis: u32) -> u32 {
    if axis == 1u {
        return params.extent.x;
    } else if axis == 2u {
        return params.extent.x + params.extent.y;
    }
    return 0u;
}

fn geometry_sample(axis: u32, coord: u32, section: u32) -> f32 {
    let span = params.extent.x + params.extent.y + params.extent.z;
    return geometry[section * span + axis_base(axis) + coord];
}

fn absorber_sample(axis: u32, coord: u32, use_half: bool) -> f32 {
    var base = axis_base(axis);
    if use_half {
        base += params.extent.x + params.extent.y + params.extent.z;
    }
    return absorber[base + coord];
}

// E[a] is half along `a` and integer along the other two, so that is exactly
// where its loss has to be sampled for the layer to stay impedance matched.
fn electric_absorption(axis: u32, coord: vec3<u32>) -> f32 {
    let b = (axis + 1u) % 3u;
    let c = (axis + 2u) % 3u;
    return absorber_sample(axis, coord[axis], true)
        + absorber_sample(b, coord[b], false)
        + absorber_sample(c, coord[c], false);
}

fn magnetic_absorption(axis: u32, coord: vec3<u32>) -> f32 {
    let b = (axis + 1u) % 3u;
    let c = (axis + 2u) % 3u;
    return absorber_sample(axis, coord[axis], false)
        + absorber_sample(b, coord[b], true)
        + absorber_sample(c, coord[c], true);
}

fn in_bounds(coord: vec3<u32>) -> bool {
    return coord.x < params.extent.x && coord.y < params.extent.y && coord.z < params.extent.z;
}

fn cell_index(coord: vec3<u32>) -> u32 {
    return (coord.z * params.extent.y + coord.y) * params.extent.x + coord.x;
}

@compute @workgroup_size(8, 8, 1)
fn update_magnetic(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = global_id;
    if !in_bounds(coord) {
        return;
    }
    let index = cell_index(coord);
    let cells = params.extent.w;
    let material = coefficients[material_index[index]];

    for (var a = 0u; a < 3u; a += 1u) {
        let b = (a + 1u) % 3u;
        let c = (a + 2u) % 3u;
        // A forward difference at the far face reaches the tangential E *on*
        // the high wall. That sample is not stored, but its value is known: a
        // perfect electric conductor pins tangential E to zero. Reading the
        // implicit zero is what makes the high faces the same PEC as the low
        // ones -- skipping these planes instead would freeze tangential H
        // half a cell inside the wall, which is a perfect *magnetic*
        // conductor: equally lossless, quietly a different boundary.
        let base_b = b * cells + index;
        let base_c = c * cells + index;
        var next_c = 0.0;
        if coord[b] + 1u < extent_of(b) {
            next_c = electric[base_c + stride_of(b)];
        }
        var next_b = 0.0;
        if coord[c] + 1u < extent_of(c) {
            next_b = electric[base_b + stride_of(c)];
        }
        let curl = geometry_sample(b, coord[b], 0u) * (next_c - electric[base_c])
            - geometry_sample(c, coord[c], 0u) * (next_b - electric[base_b]);

        let loss = material.magnetic_loss + magnetic_absorption(a, coord);
        let slot = a * cells + index;
        magnetic[slot] = ((1.0 - loss) * magnetic[slot] - material.magnetic_gain * curl)
            / (1.0 + loss);
    }
}

@compute @workgroup_size(8, 8, 1)
fn update_electric(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = global_id;
    if !in_bounds(coord) {
        return;
    }
    let index = cell_index(coord);
    let cells = params.extent.w;
    let material = coefficients[material_index[index]];

    for (var a = 0u; a < 3u; a += 1u) {
        let b = (a + 1u) % 3u;
        let c = (a + 2u) % 3u;
        // A backward difference needs the previous plane, so index 0 is never
        // written. Leaving it at zero is precisely a perfect electric
        // conductor at the low faces -- that wall is free. The high faces are
        // update_magnetic's job, where the wall's zero enters as the missing
        // forward neighbour.
        if coord[b] == 0u || coord[c] == 0u {
            continue;
        }
        let base_b = b * cells + index;
        let base_c = c * cells + index;
        let curl = geometry_sample(b, coord[b], 1u)
                * (magnetic[base_c] - magnetic[base_c - stride_of(b)])
            - geometry_sample(c, coord[c], 1u)
                * (magnetic[base_b] - magnetic[base_b - stride_of(c)]);

        let loss = material.electric_loss + electric_absorption(a, coord);
        let slot = a * cells + index;
        electric[slot] = ((1.0 - loss) * electric[slot] + material.electric_gain * curl)
            / (1.0 + loss);
    }
}

// Soft (additive) source injection. The host has already evaluated the
// waveform, so this kernel only has to place and apodize it.
@compute @workgroup_size(8, 8, 1)
fn inject(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let region = params.source_extent.xyz;
    if global_id.x >= region.x || global_id.y >= region.y || global_id.z >= region.z {
        return;
    }
    let coord = params.source_region.xyz + global_id;
    if !in_bounds(coord) {
        return;
    }

    var weight = 1.0;
    let inverse_waist_squared = params.source_shape.w;
    if inverse_waist_squared != 0.0 {
        var radius_squared = 0.0;
        for (var t = 0u; t < 3u; t += 1u) {
            // Only directions the region actually spans are apodized; a sheet
            // must not be damped along its own normal.
            if region[t] > 1u {
                // In metres, not cells: a taper stated in cells would narrow
                // wherever the grid was refined.
                let offset = geometry_sample(t, coord[t], 2u) - params.source_shape[t];
                radius_squared += offset * offset;
            }
        }
        weight = exp(-radius_squared * inverse_waist_squared);
    }

    let slot = params.source_region.w * params.extent.w + cell_index(coord);
    electric[slot] += params.source_drive.x * weight;
}
