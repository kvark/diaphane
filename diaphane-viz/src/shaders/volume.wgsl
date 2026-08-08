// Volumetric view of a live field.
//
// The fragment shader reads the solver's storage buffers directly -- the same
// allocations the compute kernels just wrote -- so a displayed frame costs no
// copies and no format conversion. That is the whole reason the solver stores
// fields as buffers rather than as textures.
//
// The default view splits the energy density into its two halves and gives
// them opposing colours. In a travelling packet they are equal and the packet
// reads as white; in a standing wave they alternate, and the volume visibly
// pulses between the two hues at twice the field frequency. That alternation
// is the electric and magnetic fields handing energy back and forth, which is
// the thing this program exists to show.

struct ViewParams {
    // nx, ny, nz, cell_count
    extent: vec4<u32>,
    // camera position in cell units, then the ray-march step in cells
    origin: vec4<f32>,
    // camera right * aspect * tan(fov/2), then exposure
    right: vec4<f32>,
    // camera up * tan(fov/2), then signed-log strength (0 = linear)
    up: vec4<f32>,
    // camera forward, then the view mode
    forward: vec4<f32>,
    // reciprocal of the reference path length, then padding
    tone: vec4<f32>,
}

const MODE_ENERGY: u32 = 0u;
const MODE_ELECTRIC: u32 = 1u;
const MODE_MAGNETIC: u32 = 2u;
const MODE_MAGNITUDE: u32 = 3u;

// Warm for electric, cool for magnetic. Distinct in hue and close in
// luminance, so neither reads as "more important" than the other.
const ELECTRIC_TINT: vec3<f32> = vec3<f32>(1.0, 0.45, 0.15);
const MAGNETIC_TINT: vec3<f32> = vec3<f32>(0.20, 0.65, 1.0);

var<storage, read> electric: array<f32>;
var<storage, read> magnetic: array<f32>;
var<storage, read_write> peak: array<atomic<u32>>;
var<uniform> view: ViewParams;

fn cell_index(coord: vec3<u32>) -> u32 {
    return (coord.z * view.extent.y + coord.y) * view.extent.x + coord.x;
}

fn field_at(base: u32, component: u32) -> vec2<f32> {
    let slot = component * view.extent.w + base;
    return vec2<f32>(electric[slot], magnetic[slot]);
}

/// Electric and magnetic energy densities at a cell, `1/2 |E|^2` and
/// `1/2 |H|^2` in normalized units where the two are directly comparable.
fn densities(coord: vec3<u32>) -> vec2<f32> {
    let base = cell_index(coord);
    var sum = vec2<f32>(0.0);
    for (var component = 0u; component < 3u; component += 1u) {
        let pair = field_at(base, component);
        sum += pair * pair;
    }
    return 0.5 * sum;
}

/// One signed component, for the diverging views.
fn signed_component(coord: vec3<u32>, component: u32) -> vec2<f32> {
    return field_at(cell_index(coord), component);
}

/// Diverging map over [-1, 1], adapted for an emissive volume.
///
/// The usual cool-warm map runs through a *light* neutral, because it is drawn
/// on white paper. Against a dark volume that neutral turns every weak,
/// uninteresting cell into grey haze that buries the wavefronts. Here the
/// neutral is dark instead: luminance carries magnitude and hue carries sign,
/// which is the pairing that works when the background is black. A naive
/// blue-through-white-to-red ramp gets this exactly backwards.
fn diverging(value: f32) -> vec3<f32> {
    let neutral = vec3<f32>(0.05, 0.05, 0.07);
    let cold = vec3<f32>(0.25, 0.48, 1.0);
    let hot = vec3<f32>(1.0, 0.30, 0.18);
    let magnitude = min(abs(value), 1.0);
    if value < 0.0 {
        return mix(neutral, cold, magnitude);
    }
    return mix(neutral, hot, magnitude);
}

/// `sign(v) * log(1 + k|v|) / log(1 + k)`.
///
/// Compresses the dynamic range so a weak scattered field is visible in the
/// same frame as the source that produced it. Linear scaling alone makes most
/// of the interesting physics invisible.
fn signed_log(value: f32, strength: f32) -> f32 {
    if strength <= 0.0 {
        return value;
    }
    return sign(value) * log(1.0 + strength * abs(value)) / log(1.0 + strength);
}

/// Slab-method intersection with the domain box, in cell units.
fn intersect_box(origin: vec3<f32>, direction: vec3<f32>, size: vec3<f32>) -> vec2<f32> {
    let inverse = 1.0 / direction;
    let low = (vec3<f32>(0.0) - origin) * inverse;
    let high = (size - origin) * inverse;
    let near = min(low, high);
    let far = max(low, high);
    return vec2<f32>(
        max(max(near.x, near.y), near.z),
        min(min(far.x, far.y), far.z),
    );
}

/// A faint outline where a ray leaves the box close to two of its faces, so
/// the volume reads as a solid in space rather than as a floating haze.
fn edge_glow(point: vec3<f32>, size: vec3<f32>) -> f32 {
    let distance = min(point, size - point) / size;
    var closest = vec3<f32>(distance.x, distance.y, distance.z);
    // Second-smallest of the three normalized face distances: small only near
    // an edge, not merely near a face.
    let sorted_min = min(min(closest.x, closest.y), closest.z);
    let sorted_max = max(max(closest.x, closest.y), closest.z);
    let middle = closest.x + closest.y + closest.z - sorted_min - sorted_max;
    return smoothstep(0.01, 0.0, middle);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen: vec2<f32>,
}

@vertex
fn main_vs(@builtin(vertex_index) index: u32) -> VertexOutput {
    // Fullscreen triangle: cheaper than a quad and free of the diagonal seam.
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    let ndc = corner * 2.0 - 1.0;
    var out: VertexOutput;
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    // Clip space has y downwards; the camera basis does not.
    out.screen = vec2<f32>(ndc.x, -ndc.y);
    return out;
}

@fragment
fn main_fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let size = vec3<f32>(view.extent.xyz);
    let origin = view.origin.xyz;
    let direction = normalize(
        view.forward.xyz + input.screen.x * view.right.xyz + input.screen.y * view.up.xyz
    );
    let sky = background(input.screen);

    let span = intersect_box(origin, direction, size);
    let enter = max(span.x, 0.0);
    if span.y <= enter {
        return vec4<f32>(sky, 1.0);
    }

    let march = max(view.origin.w, 0.25);
    let exposure = view.right.w;
    let log_strength = view.up.w;
    let mode = u32(view.forward.w);
    let signed_view = mode == MODE_ELECTRIC || mode == MODE_MAGNETIC;

    // Energy densities are non-negative, so integrating them along the ray is
    // a physically meaningful line integral and the volume glows.
    //
    // A signed field component is not: a ray crossing half a wavelength sees
    // equal and opposite lobes, and integrating them -- with any colour map --
    // averages the wave away into a uniform haze. So the signed views take the
    // extreme sample along the ray instead of the sum, which is a maximum
    // intensity projection and keeps the wavefronts crisp.
    var glow = vec3<f32>(0.0);
    var extreme = 0.0;
    var distance = enter + 0.5 * march;
    // Bounded so a grazing ray through a large domain cannot stall the frame.
    for (var taken = 0u; taken < 4096u; taken += 1u) {
        if distance >= span.y {
            break;
        }
        let point = origin + direction * distance;
        let coord = clamp(
            vec3<u32>(max(point, vec3<f32>(0.0))),
            vec3<u32>(0u),
            view.extent.xyz - 1u,
        );
        if signed_view {
            let value = component_at(coord, mode) * exposure;
            if abs(value) > abs(extreme) {
                extreme = value;
            }
        } else {
            glow += emission(coord, mode, exposure, log_strength);
        }
        distance += march;
    }

    let rim = edge_glow(origin + direction * span.y, size) * vec3<f32>(0.16, 0.18, 0.22);

    if signed_view {
        let scaled = signed_log(extreme, log_strength);
        let opacity = min(abs(scaled), 1.0);
        return vec4<f32>(diverging(scaled) * opacity + sky * (1.0 - opacity) + rim, 1.0);
    }

    // Divide by a reference length, or every ray through the domain saturates
    // and the volume reads as one white blob.
    glow *= march * view.tone.x;
    // Fold the unbounded sum back into [0, 1) rather than clipping, so a bright
    // core keeps its shape instead of flattening.
    let opacity = vec3<f32>(1.0) - exp(-glow);
    return vec4<f32>(opacity + sky * exp(-glow) + rim, 1.0);
}

/// Emitted colour per unit length at a cell, for the additive views.
fn emission(coord: vec3<u32>, mode: u32, exposure: f32, log_strength: f32) -> vec3<f32> {
    let pair = densities(coord) * exposure;
    if mode == MODE_MAGNITUDE {
        return vec3<f32>(signed_log(pair.x + pair.y, log_strength));
    }
    // Electric warm, magnetic cool. Their tints sum to roughly white, so a
    // travelling packet -- where the two are equal -- reads as white, and any
    // departure from equipartition shows up directly as a colour cast.
    return signed_log(pair.x, log_strength) * ELECTRIC_TINT
        + signed_log(pair.y, log_strength) * MAGNETIC_TINT;
}

/// The signed component a diverging view is showing: `Ez` or `Hz`.
fn component_at(coord: vec3<u32>, mode: u32) -> f32 {
    let pair = signed_component(coord, 2u);
    if mode == MODE_MAGNETIC {
        return pair.y;
    }
    return pair.x;
}

fn background(screen: vec2<f32>) -> vec3<f32> {
    // A barely-there vertical gradient: enough to tell the volume from the
    // void without competing with it.
    let shade = 0.5 + 0.5 * screen.y;
    return mix(vec3<f32>(0.02, 0.02, 0.03), vec3<f32>(0.05, 0.05, 0.07), shade);
}

// Auto-ranging. One dispatch reduces the loudest energy density in the domain
// into a single slot with an atomic max on the float's bit pattern, which is
// monotonic for non-negative values. The host reads it back and smooths it,
// which is why the display does not flicker as a pulse grows and decays across
// several orders of magnitude.
@compute @workgroup_size(8, 8, 1)
fn measure_peak(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id >= view.extent.xyz) {
        return;
    }
    let pair = densities(global_id);
    atomicMax(&peak[0], bitcast<u32>(max(pair.x, pair.y)));
}
