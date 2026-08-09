//! The scene file format, and the curated scenes in `scenes/`.
//!
//! Two jobs. The round-trip tests make sure the format can express everything
//! a scene can be — including the `inf` conductivity of a perfect conductor,
//! which is exactly the value a text format is most likely to mangle. The
//! directory test makes sure the checked-in example scenes still parse and
//! still describe a resolvable problem, so they cannot rot silently while the
//! types around them change.

use diaphane::{
    Axis, Extent, Material, Refinement, Scene, Shape, Source, Waveform, cpu, grid::Grid,
};
use std::{fs, path::PathBuf};

fn scene_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scenes")
}

fn scene_files() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(scene_directory())
        .expect("scenes/ exists")
        .map(|entry| entry.expect("readable directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "ron"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "scenes/ has no .ron files");
    files
}

#[test]
fn every_preset_survives_a_round_trip() {
    let presets = [
        ("photon", Scene::photon(Extent::cube(48))),
        ("cavity", Scene::cavity(Extent::cube(48))),
        ("slab", Scene::slab(Extent::cube(48), 1.8)),
    ];
    for (name, scene) in presets {
        let text = scene.to_ron().unwrap_or_else(|e| panic!("{name}: {e}"));
        let parsed = Scene::from_ron(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(parsed, scene, "{name} changed on the way through RON");
    }
}

#[test]
fn an_infinite_conductivity_survives_a_round_trip() {
    // A perfect conductor is the σ → ∞ limit rather than a flag, so the format
    // has to carry a literal infinity. Most text formats quietly turn it into
    // null, a parse error, or the largest finite float — and the last one
    // would be the worst, because the scene would still load and the metal
    // would still look like metal while no longer being one.
    let mut scene = Scene::sized([0.024; 3], 1000.0);
    let metal = scene.materials.push(Material::PERFECT_CONDUCTOR);
    scene.shapes.push(Shape::Sphere {
        center: [0.0; 3],
        radius: 0.006,
        material: metal,
    });

    let parsed = Scene::from_ron(&scene.to_ron().unwrap()).unwrap();
    assert_eq!(parsed, scene);
    assert!(parsed.materials.get(metal).conductivity.is_infinite());

    // And it still behaves like a conductor after the trip.
    let coefficients = parsed.materials.get(metal).coefficients(&parsed.grid);
    assert_eq!(coefficients.electric_gain, 0.0);
    assert_eq!(coefficients.electric_loss, 1.0);
}

#[test]
fn the_curated_scenes_parse_and_are_resolvable() {
    for path in scene_files() {
        let scene = Scene::load(&path).unwrap_or_else(|e| panic!("{e}"));
        scene
            .validate()
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(
            !scene.sources.is_empty(),
            "{}: a scene with no source does nothing",
            path.display()
        );
    }
}

#[test]
fn the_curated_scenes_actually_paint_their_geometry() {
    // A shape in the wrong units paints nothing, and a scene that renders an
    // empty box is the quietest possible failure. Any scene declaring a
    // material beyond vacuum has to actually use it.
    for path in scene_files() {
        let scene = Scene::load(&path).unwrap_or_else(|e| panic!("{e}"));
        if scene.materials.len() == 1 {
            assert!(
                scene.shapes.is_empty(),
                "{}: shapes but only vacuum in the table",
                path.display()
            );
            continue;
        }
        let indices = scene.material_indices();
        for material in 1..scene.materials.len() as u32 {
            let painted = indices.iter().filter(|&&i| i == material).count();
            assert!(
                painted > 0,
                "{}: material {material} is declared but paints no cells",
                path.display()
            );
        }
    }
}

#[test]
fn the_curated_scenes_run() {
    // Cheap, but it catches a scene that parses and validates and then panics
    // on construction — the class of mistake `validate` cannot see coming.
    for path in scene_files() {
        let scene = Scene::load(&path).unwrap_or_else(|e| panic!("{e}"));
        let mut simulation = cpu::Simulation::new(&scene.with_resolution(400.0));
        simulation.advance_by(30);
        assert!(
            simulation.is_finite(),
            "{}: diverged within 30 steps",
            path.display()
        );
        assert!(
            simulation.energy().total() > 0.0,
            "{}: the source deposited nothing",
            path.display()
        );
    }
}

#[test]
fn a_saved_scene_reloads_identically() {
    let scene = Scene::slab(Extent::cube(32), 2.0);
    let path = std::env::temp_dir().join("diaphane-scene-round-trip.ron");
    scene.save(&path).unwrap();
    assert_eq!(Scene::load(&path).unwrap(), scene);
    let _ = fs::remove_file(&path);
}

#[test]
fn a_syntax_error_is_reported_rather_than_panicking() {
    let error = Scene::from_ron("Scene(grid: Grid(extent: Extent(x: 4").unwrap_err();
    assert!(!error.is_empty(), "the parse error carried no message");
}

#[test]
fn a_graded_scene_survives_a_round_trip() {
    // The preset round trips above all ride uniform grids, which cannot see a
    // mistake in the graded recipe: `Grid` serializes as the `GridSpec` that
    // built it, so dropping, say, `max_ratio` on the way out would quietly
    // remesh every saved scene on reload. Scene equality includes the resolved
    // spacing, so this asserts the rebuilt mesh, not merely the recipe.
    let grid = Grid::graded_at(
        [0.048, 0.032, 0.02],
        1e3,
        vec![Refinement::across(Axis::X, 0.006, 0.008, 0.25e-3)],
        1.05,
    );
    let mut scene = Scene::on_grid(grid);
    let frequency = scene.grid.frequency_for_resolution(24.0);
    let film = scene.materials.push(Material::refractive(2.0));
    scene.shapes.push(Shape::Slab {
        axis: Axis::X,
        offset: 0.006,
        thickness: 0.002,
        material: film,
    });
    let scene = scene.with_source(Source::point(
        [0.0; 3],
        Axis::Z,
        Waveform::ricker(frequency),
    ));
    // The grading engaged: more cells along x than the coarse recipe asks for.
    assert!(scene.grid.extent.x > 48);

    let text = scene.to_ron().unwrap();
    let reloaded = Scene::from_ron(&text).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(reloaded, scene);
}

#[test]
fn a_zero_cell_size_is_reported_rather_than_panicking() {
    // Not a syntax error: the file parses, and the mistake used to detonate
    // an assert inside deserialization instead of coming back as the `Err`
    // that `from_ron` promises.
    let text = Scene::slab(Extent::cube(24), 1.8).to_ron().unwrap();
    assert!(text.contains("cell_size: 0.001"), "{text}");
    let error = Scene::from_ron(&text.replace("cell_size: 0.001", "cell_size: 0.0")).unwrap_err();
    assert!(error.contains("cell_size"), "{error}");
}
