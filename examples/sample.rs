use {racist::*, rand_xorshift::XorShiftRng};

fn main() -> Result<(), image::error::ImageError> {
    let camera = Camera::new(Fov::deg(45.0), (800, 600)).at(0.0, 0.0, 5.0);

    let mut scene = Scene::new();

    // materials
    let white = scene.material(Material::diffuse([1.0, 1.0, 1.0]));
    let green_mirror = scene.material(Material::mirror([0.25, 1.0, 0.25]));
    // let red = scene.material(Diffuse::rgb(1.0, 0.25, 0.25));
    // let blue = scene.material(Diffuse::rgb(0.25, 0.25, 1.0));
    // let water = scene.material(Dielectric::transparent(1.33));
    // let crown_glass = scene.material(Dielectric::transparent(1.52));
    // let diamond = scene.material(Dielectric::transparent(2.417));
    let white_light = scene.material(Material::light([1.0, 1.0, 1.0], 50.0));

    // lights
    scene.object(Sphere::new(0.0, 3.0, -3.0).scaled(0.1), white_light);

    // objects
    scene
        .object(Sphere::new(-2.5, 0.5, -3.0), green_mirror)
        .object(Sphere::new(1.0, -1.0, -2.5).scaled(0.5), white);
    // scene.add_object(Sphere::unit_at(2.0, 0.5, -5.0).scaled(1.5), red);
    // scene.add_object(Sphere::unit_at(-2.0, 2.0, -6.0).scaled(2.0), blue);
    // scene.add_object(Sphere::unit_at(-1.0, -0.5, -2.5).scaled(0.5), water);
    // scene.add_object(Sphere::unit_at(0.0, -0.75, -2.5).scaled(0.5), crown_glass);
    // scene.add_object(Sphere::unit_at(1.0, -1.0, -2.5).scaled(0.5), diamond);

    // surrounding box
    // scene.add_object(Plane::facing_pos_x().shifted_back(5.0), red); // LEFT
    // scene.add_object(Plane::facing_neg_x().shifted_back(5.0), blue); // RIGHT
    // scene.add_object(Plane::facing_pos_y().shifted_back(2.0), white); // BOTTOM
    // scene.add_object(Plane::facing_neg_y().shifted_back(4.0), white); // TOP
    // scene.add_object(Plane::facing_pos_z().shifted_back(7.0), white); // FRONT
    // scene.add_object(Plane::facing_neg_z().shifted_back(7.0), white); // BACK

    let tracer = PathTracer { depth: 10 };
    render::<PathTracer, f32, XorShiftRng>(tracer, scene, camera, 8800).save("spheres.png")?;

    Ok(())
}
