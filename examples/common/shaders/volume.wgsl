// Volumetric view of a live field.
//
// The fragment shader reads the solver's storage buffers directly -- the same
// allocations the compute kernels just wrote -- so a displayed frame costs no
// copies and no format conversion. That is the whole reason the solver stores
// fields as buffers rather than as textures.
//
// The default view shows signed E and signed H at once, in two hues, each
// taken along the component that actually carries the field -- which is not
// the same component for both. A wave along x polarized in z has its magnetic
// field in y, so pairing Ez with Hz shows one real field and one that is a
// hundredth of it. The host picks both indices from a reduction rather than
// assuming, because the answer depends on the scene.
//
// What the two hues do *not* show is orthogonality. E and H are perpendicular
// as vectors, but in a travelling wave they are co-located and in phase, so
// any pair of scalars drawn from them lands on the same sheets. They separate
// in a standing wave, where they sit a quarter wavelength apart -- so the
// cavity, not free flight, is where the fields visibly trade energy.

struct ViewParams {
    // nx, ny, nz, cell_count
    extent: vec4<u32>,
    // domain size in coarse cells, then the lookup sample count per axis
    box_size: vec4<f32>,
    // camera position in cell units, then the ray-march step in cells
    origin: vec4<f32>,
    // camera right * aspect * tan(fov/2), then exposure
    right: vec4<f32>,
    // camera up * tan(fov/2), then signed-log strength (0 = linear)
    up: vec4<f32>,
    // camera forward, then the view mode
    forward: vec4<f32>,
    // reciprocal of the reference path length, then the scrub bar:
    // played fraction, keyframed-window start fraction, bar height in
    // normalized screen units (0 hides it)
    tone: vec4<f32>,
    // component of E and of H the signed views should read, then padding
    components: vec4<u32>,
}

const MODE_ENERGY: u32 = 0u;
const MODE_ELECTRIC: u32 = 1u;
const MODE_MAGNETIC: u32 = 2u;
const MODE_MAGNITUDE: u32 = 3u;
// The mesh itself, with no field in it: emission at every cell boundary, so
// the lines bunch up wherever the grid is refined and the graded regions read
// as brighter. The one view whose subject is the discretization.
const MODE_GRID: u32 = 5u;
// Signed Ez and Hz at once, in two hues. The default, because it is the only
// view in which the two fields are separately visible: the energy densities of
// a travelling wave are *equal* (that is equipartition, and there is a test for
// it), so tinting them warm and cool sums to white and shows a colourless blob.
// The signed components still swing through zero as the packet passes, so this
// shows wavefronts -- and shows them locked in phase, which is what a
// travelling wave does. Run the cavity to see the two actually alternate.
const MODE_FIELDS: u32 = 4u;

// Warm for electric, cool for magnetic. Distinct in hue and close in
// luminance, so neither reads as "more important" than the other.
const ELECTRIC_TINT: vec3<f32> = vec3<f32>(1.0, 0.45, 0.15);
const MAGNETIC_TINT: vec3<f32> = vec3<f32>(0.20, 0.65, 1.0);

var<storage, read> electric: array<f32>;
var<storage, read> magnetic: array<f32>;
var<storage, read_write> peak: array<atomic<u32>>;
// Inverse of the cumulative cell width: three sections of `box_size.w`
// samples, each giving a fractional cell coordinate. See `Grid::cell_lookup`.
var<storage, read> lookup: array<f32>;
var<uniform> view: ViewParams;

/// World position, in coarse cells, to a fractional cell coordinate.
///
/// The identity on a uniform grid. On a graded one the two spaces differ --
/// cell indices stretch wherever the mesh is fine -- and marching in index
/// space would render the refined region several times its actual size. A
/// tabulated inverse keeps that a pair of loads rather than a search.
fn cell_of(point: vec3<f32>) -> vec3<f32> {
    let samples = u32(view.box_size.w);
    var out = vec3<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let fraction = clamp(point[axis] / view.box_size[axis], 0.0, 1.0);
        let position = fraction * f32(samples - 1u);
        let low = u32(position);
        let high = min(low + 1u, samples - 1u);
        let base = axis * samples;
        out[axis] = mix(lookup[base + low], lookup[base + high], position - f32(low));
    }
    return out;
}

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

/// Slab-method intersection with the domain box, in coarse-cell units.
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
    // Passed straight through: blade flips the viewport, so +y in clip space
    // is already up on screen. Negating here -- the reflex, because Vulkan
    // clip space is nominally y-down -- mirrors the whole render vertically,
    // which on a symmetric wave packet in a cube looks like nothing at all.
    out.screen = ndc;
    return out;
}

@fragment
fn main_fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let size = view.box_size.xyz;
    let origin = view.origin.xyz;
    let direction = normalize(
        view.forward.xyz + input.screen.x * view.right.xyz + input.screen.y * view.up.xyz
    );
    let sky = background(input.screen);

    let span = intersect_box(origin, direction, size);
    let enter = max(span.x, 0.0);
    if span.y <= enter {
        // Still composite the bar: most of the bottom edge is rays that miss
        // the volume entirely, so returning early here drew the bar only
        // across the width of the box.
        return vec4<f32>(overlay(sky, input.screen), 1.0);
    }

    let march = max(view.origin.w, 0.25);
    let exposure = view.right.w;
    let log_strength = view.up.w;
    let mode = u32(view.forward.w);
    let signed_view = mode == MODE_ELECTRIC || mode == MODE_MAGNETIC || mode == MODE_FIELDS;

    // Energy densities are non-negative, so integrating them along the ray is
    // a physically meaningful line integral and the volume glows.
    //
    // A signed field component is not: a ray crossing half a wavelength sees
    // equal and opposite lobes, and integrating them -- with any colour map --
    // averages the wave away into a uniform haze. So the signed views take the
    // extreme sample along the ray instead of the sum, which is a maximum
    // intensity projection and keeps the wavefronts crisp.
    var glow = vec3<f32>(0.0);
    // Two extremes, so `MODE_FIELDS` can keep E and H apart; the single-field
    // views leave the second at zero.
    var extreme = vec2<f32>(0.0);
    var distance = enter + 0.5 * march;
    // Bounded so a grazing ray through a large domain cannot stall the frame.
    for (var taken = 0u; taken < 4096u; taken += 1u) {
        if distance >= span.y {
            break;
        }
        let point = origin + direction * distance;
        let fractional = cell_of(point);
        let coord = clamp(
            vec3<u32>(max(fractional, vec3<f32>(0.0))),
            vec3<u32>(0u),
            view.extent.xyz - 1u,
        );
        if mode == MODE_GRID {
            // Distance to the nearest cell boundary, in cells -- so a line is
            // a fixed fraction of whatever cell it bounds, and refined regions
            // pack more of them into the same distance. Accumulating along the
            // ray then makes those regions brighter, which is the point: this
            // view answers "where did the resolution go".
            let edge = abs(fractional - round(fractional));
            let nearest = min(min(edge.x, edge.y), edge.z);
            glow += GRID_TINT * smoothstep(0.08, 0.0, nearest);
        } else if signed_view {
            let value = signed_pair(coord, mode) * exposure;
            if abs(value.x) > abs(extreme.x) {
                extreme.x = value.x;
            }
            if abs(value.y) > abs(extreme.y) {
                extreme.y = value.y;
            }
        } else {
            glow += emission(coord, mode, exposure, log_strength);
        }
        distance += march;
    }

    let rim = edge_glow(origin + direction * span.y, size) * vec3<f32>(0.16, 0.18, 0.22);

    if mode == MODE_FIELDS {
        // Each field gets its own hue and its own sign, so a positive Ez sheet
        // and a positive Hz sheet are different colours and a sign flip is a
        // different brightness rather than a different colour -- signs matter
        // less than which field you are looking at.
        let e = signed_log(extreme.x, log_strength);
        let h = signed_log(extreme.y, log_strength);
        let lit = abs(e) * ELECTRIC_TINT + abs(h) * MAGNETIC_TINT;
        let shade = 0.5 + 0.5 * vec3<f32>(sign(e) * abs(e), 0.0, sign(h) * abs(h));
        let field = sky * exp(-length(lit)) + lit * shade + rim;
        return vec4<f32>(overlay(field, input.screen), 1.0);
    }
    if signed_view {
        let scaled = signed_log(extreme.x, log_strength);
        let opacity = min(abs(scaled), 1.0);
        let field = diverging(scaled) * opacity + sky * (1.0 - opacity) + rim;
        return vec4<f32>(overlay(field, input.screen), 1.0);
    }

    // Divide by a reference length, or every ray through the domain saturates
    // and the volume reads as one white blob.
    glow *= march * view.tone.x;
    // Fold the unbounded sum back into [0, 1) rather than clipping, so a bright
    // core keeps its shape instead of flattening.
    let opacity = vec3<f32>(1.0) - exp(-glow);
    let field = opacity + sky * exp(-glow) + rim;
    return vec4<f32>(overlay(field, input.screen), 1.0);
}

fn overlay(field: vec3<f32>, screen: vec2<f32>) -> vec3<f32> {
    let bar = scrub_bar(screen);
    return mix(field, bar.rgb, bar.a);
}

/// Colour of a cell boundary in [`MODE_GRID`]. Dim, because a ray crosses many.
const GRID_TINT: vec3<f32> = vec3<f32>(0.16, 0.19, 0.26);

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

/// The signed components a diverging view is showing.
///
/// `x` carries whatever the mode's primary field is; `y` is the magnetic one
/// and is zero unless both are wanted at once.
fn signed_pair(coord: vec3<u32>, mode: u32) -> vec2<f32> {
    let base = cell_index(coord);
    let electric = field_at(base, view.components.x).x;
    let magnetic = field_at(base, view.components.y).y;
    if mode == MODE_FIELDS {
        return vec2<f32>(electric, magnetic);
    }
    if mode == MODE_MAGNETIC {
        return vec2<f32>(magnetic, 0.0);
    }
    return vec2<f32>(electric, 0.0);
}

/// A scrub bar along the bottom edge.
///
/// Drawn here rather than composited afterwards because the renderer has no
/// other surface to draw on -- there is no UI layer -- and a slider you can see
/// the extent of is what makes "how far back can I drag" answerable without
/// documentation. Returns a colour to blend over the frame, and zero alpha
/// everywhere else.
fn scrub_bar(screen: vec2<f32>) -> vec4<f32> {
    let height = view.tone.w;
    if height <= 0.0 {
        return vec4<f32>(0.0);
    }
    // Both this and the host's hit test measure from the bottom edge as a
    // fraction of the *image*, so `screen.y` (which spans -1 to 1) is halved.
    // Leaving it unhalved makes the region the pointer scrubs in twice the
    // region that is drawn, which reads as the bar responding above itself.
    let from_bottom = 0.5 * (screen.y + 1.0);
    if from_bottom > height {
        return vec4<f32>(0.0);
    }

    let played = view.tone.y;
    let window_start = view.tone.z;
    let position = 0.5 * (screen.x + 1.0);

    let track = vec3<f32>(0.12, 0.13, 0.16);
    // The span that can be scrubbed instantly, because keyframes cover it.
    let windowed = vec3<f32>(0.22, 0.30, 0.40);
    let elapsed = vec3<f32>(0.35, 0.62, 0.95);
    let head = vec3<f32>(1.0, 0.85, 0.45);

    var colour = track;
    if position >= window_start {
        colour = windowed;
    }
    if position <= played {
        colour = elapsed;
    }
    // A bright playhead, two pixels wide at any sensible resolution.
    if abs(position - played) < 0.0025 {
        colour = head;
    }
    // Fade the top edge so the bar does not read as a hard crop of the scene.
    let softness = smoothstep(height, height * 0.6, from_bottom);
    return vec4<f32>(colour, 0.55 + 0.45 * softness);
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
    // Per-component peaks too, so the host can pair each field with the
    // component that carries it instead of guessing. `atomicMax` on the bit
    // pattern is an ordinary max for non-negative floats: IEEE-754 orders them
    // the same way the integers do.
    let base = cell_index(global_id);
    for (var component = 0u; component < 3u; component += 1u) {
        let both = abs(field_at(base, component));
        atomicMax(&peak[1u + component], bitcast<u32>(both.x));
        atomicMax(&peak[4u + component], bitcast<u32>(both.y));
    }
}
