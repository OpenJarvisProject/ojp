//! Typeset LaTeX display math ($$ ... $$ and \\[ ... \\]) as images with
//! RaTeX (pure Rust, KaTeX-compatible, fonts embedded). Inline math stays
//! Unicode text (format.rs): an image in the middle of a line can't flow
//! with the words around it.
use ratex_layout::{LayoutOptions, layout, to_display_list, validate_layout};
use ratex_parser::parser::parse;
use ratex_render::{RenderOptions, render_to_png};
use ratex_types::color::Color;
use ratex_types::math_style::MathStyle;

#[derive(Debug, PartialEq, Eq)]
pub enum Segment {
    Text(String),
    /// LaTeX source of one display formula.
    Display(String),
}

/// Split `text` at complete display-math blocks. An unclosed opener is left
/// in the text (it may still be streaming in).
pub fn split(text: &str) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let found = [("$$", "$$"), ("\\[", "\\]")]
            .iter()
            .filter_map(|&(open, close)| rest.find(open).map(|i| (i, open, close)))
            .min_by_key(|&(i, _, _)| i);
        let Some((start, open, close)) = found else { break };
        let body_start = start + open.len();
        let Some(len) = rest[body_start..].find(close) else { break };
        let body = rest[body_start..body_start + len].trim();
        if !body.is_empty() {
            if start > 0 {
                out.push(Segment::Text(rest[..start].to_string()));
            }
            out.push(Segment::Display(body.to_string()));
        } else if start > 0 {
            out.push(Segment::Text(rest[..start].to_string()));
        }
        rest = &rest[body_start + len + close.len()..];
    }
    if !rest.is_empty() || out.is_empty() {
        out.push(Segment::Text(rest.to_string()));
    }
    out
}

/// A rendered formula: straight-alpha RGBA, black ink on transparent
/// (tint it with the theme colour), at twice the logical size.
pub struct Rendered {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Typeset `latex` at `font_px` logical pixels per em; None if it doesn't parse or lay out.
pub fn render(latex: &str, font_px: f32) -> Option<Rendered> {
    let ast = parse(latex).ok()?;
    let options = LayoutOptions::default().with_style(MathStyle::Display).with_color(Color::BLACK);
    let lbox = layout(&ast, &options);
    validate_layout(&lbox).ok()?;
    let list = to_display_list(&lbox);
    let png_bytes = render_to_png(
        &list,
        &RenderOptions {
            font_size: font_px,
            padding: 4.0,
            background_color: Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
            font_dir: String::new(),
            device_pixel_ratio: 2.0,
        },
    )
    .ok()?;
    let mut reader = png::Decoder::new(png_bytes.as_slice()).read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => buf.chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        _ => return None,
    };
    Some(Rendered { width: info.width, height: info.height, rgba })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_display_blocks_only() {
        assert_eq!(
            split(r"So $$x^2$$ and \( y \) then \[ z \] done"),
            vec![
                Segment::Text("So ".into()),
                Segment::Display("x^2".into()),
                Segment::Text(r" and \( y \) then ".into()),
                Segment::Display("z".into()),
                Segment::Text(" done".into()),
            ]
        );
        assert_eq!(split("plain"), vec![Segment::Text("plain".into())]);
        assert_eq!(split("open $$ x^2"), vec![Segment::Text("open $$ x^2".into())]);
    }

    #[test]
    fn renders_a_fraction() {
        let r = render(r"\frac{a+b}{c} = \sqrt{x^2}", 18.0).expect("renders");
        assert!(r.width > 20 && r.height > 20);
        assert_eq!(r.rgba.len(), (r.width * r.height * 4) as usize);
        assert!(r.rgba.chunks(4).any(|p| p[3] > 200), "has ink");
        assert!(render(r"\frac{", 18.0).is_none());
    }
}
