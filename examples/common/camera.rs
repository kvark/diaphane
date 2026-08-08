//! An orbit camera, in coarse-cell units.

use std::f32;

/// Orbits the centre of the domain.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    /// Distance from the centre, as a multiple of the domain's longest side.
    pub distance: f32,
    /// Vertical field of view, radians.
    pub fov: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            yaw: 0.9,
            pitch: 0.45,
            distance: 1.9,
            fov: 0.8,
        }
    }

    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        // Stop just short of the poles, where the up vector degenerates.
        const LIMIT: f32 = 0.5 * f32::consts::PI - 0.01;
        self.pitch = (self.pitch + delta_pitch).clamp(-LIMIT, LIMIT);
    }

    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(0.6, 8.0);
    }

    /// Position and the three basis vectors the ray-march wants, already
    /// scaled by the field of view and aspect ratio so the shader only has to
    /// add them up.
    ///
    /// `size` is the domain in coarse cells -- [`Grid::box_size`] -- not the
    /// cell counts. The two agree on a uniform grid and diverge on a graded
    /// one, where cell indices stretch wherever the mesh is fine and a camera
    /// framed on them would frame a shape the domain does not have.
    pub fn basis(&self, size: [f32; 3], aspect: f32) -> CameraBasis {
        let center = size.map(|v| 0.5 * v);
        let radius = size[0].max(size[1]).max(size[2]);

        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let away = [cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw];
        let position = std::array::from_fn(|a| center[a] + away[a] * self.distance * radius);
        let forward = away.map(|v| -v);

        let world_up = [0.0, 1.0, 0.0];
        let right = normalize(cross(forward, world_up));
        let up = cross(right, forward);

        let tangent = (0.5 * self.fov).tan();
        CameraBasis {
            position,
            right: right.map(|v| v * tangent * aspect),
            up: up.map(|v| v * tangent),
            forward,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CameraBasis {
    pub position: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|c| c / norm)
}

#[cfg(test)]
mod tests {
    use super::Camera;

    #[test]
    fn looks_at_the_centre_of_the_domain() {
        let basis = Camera::new().basis([40.0, 60.0, 80.0], 1.0);
        // Walking forward from the camera by the orbit distance must land on
        // the domain centre.
        let radius = 80.0 * 1.9;
        let landing: Vec<f32> = (0..3)
            .map(|a| basis.position[a] + basis.forward[a] * radius)
            .collect();
        for (axis, &expected) in [20.0, 30.0, 40.0].iter().enumerate() {
            assert!(
                (landing[axis] - expected).abs() < 1e-3,
                "axis {axis}: {} vs {expected}",
                landing[axis]
            );
        }
    }

    #[test]
    fn the_basis_stays_orthogonal_while_orbiting() {
        let mut camera = Camera::new();
        for _ in 0..40 {
            camera.orbit(0.3, 0.2);
            let basis = camera.basis([32.0; 3], 1.6);
            let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            assert!(dot(basis.right, basis.forward).abs() < 1e-4);
            assert!(dot(basis.up, basis.forward).abs() < 1e-4);
            assert!(dot(basis.right, basis.up).abs() < 1e-4);
        }
    }

    #[test]
    fn pitch_never_reaches_the_pole() {
        let mut camera = Camera::new();
        for _ in 0..50 {
            camera.orbit(0.0, 1.0);
        }
        assert!(camera.pitch < 0.5 * std::f32::consts::PI);
        for _ in 0..100 {
            camera.orbit(0.0, -1.0);
        }
        assert!(camera.pitch > -0.5 * std::f32::consts::PI);
    }

    #[test]
    fn zoom_is_bounded() {
        let mut camera = Camera::new();
        for _ in 0..100 {
            camera.zoom(0.5);
        }
        assert!(camera.distance >= 0.6);
        for _ in 0..100 {
            camera.zoom(2.0);
        }
        assert!(camera.distance <= 8.0);
    }
}
