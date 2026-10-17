//! Typeset one formula to a PNG: ... --example math_png -- OUT.png 'latex'
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = ojp_core::math::render(&a[2], 18.0).expect("formula didn't typeset");
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, r.width, r.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&r.rgba).unwrap();
    }
    std::fs::write(&a[1], out).unwrap();
    println!("{}x{}", r.width, r.height);
}
