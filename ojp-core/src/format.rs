//! Make model output readable in a plain chat bubble.
//!
//! Models write math as LaTeX (`\( 2 \times 3 = 6 \)`, `\frac{1}{2}`,
//! `x^2`). The UI renders markdown but has no math engine, so this turns
//! the common LaTeX into Unicode text instead: `2 × 3 = 6`, `1/2`, `x²`.
//! Anything it doesn't recognise is left as written, minus the delimiters.

/// Convert LaTeX math (inline `\( \)`, `$ $`, display `\[ \]`, `$$ $$`)
/// to Unicode; text outside math is returned unchanged.
pub fn latex_to_unicode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        let next = [("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$"), ("$", "$")]
            .iter()
            .filter_map(|&(open, close)| rest.find(open).map(|i| (i, open, close)))
            .min_by_key(|&(i, open, _)| (i, usize::MAX - open.len()));
        let Some((start, open, close)) = next else {
            out.push_str(rest);
            break;
        };
        let body_start = start + open.len();
        let Some(len) = rest[body_start..].find(close) else {
            out.push_str(rest);
            break;
        };
        let body = &rest[body_start..body_start + len];
        // A lone "$" followed by a digit and a space is usually money
        // ("costs $5 and $7"), not math.
        let looks_like_money = open == "$" && (body.starts_with(' ') || body.ends_with(' '));
        out.push_str(&rest[..start]);
        if looks_like_money {
            out.push_str(&rest[start..body_start + len]);
            rest = &rest[body_start + len..];
            continue;
        }
        out.push_str(&convert_math(body));
        rest = &rest[body_start + len + close.len()..];
    }
    out
}

const SYMBOLS: &[(&str, &str)] = &[
    ("times", "×"), ("div", "÷"), ("cdot", "·"), ("pm", "±"), ("mp", "∓"),
    ("leq", "≤"), ("le", "≤"), ("geq", "≥"), ("ge", "≥"), ("neq", "≠"), ("ne", "≠"),
    ("approx", "≈"), ("equiv", "≡"), ("infty", "∞"), ("degree", "°"), ("circ", "°"),
    ("rightarrow", "→"), ("to", "→"), ("leftarrow", "←"), ("Rightarrow", "⇒"),
    ("implies", "⇒"), ("sum", "∑"), ("prod", "∏"), ("int", "∫"), ("partial", "∂"),
    ("in", "∈"), ("notin", "∉"), ("subset", "⊂"), ("cup", "∪"), ("cap", "∩"),
    ("forall", "∀"), ("exists", "∃"), ("emptyset", "∅"), ("angle", "∠"),
    ("alpha", "α"), ("beta", "β"), ("gamma", "γ"), ("delta", "δ"), ("epsilon", "ε"),
    ("theta", "θ"), ("lambda", "λ"), ("mu", "μ"), ("pi", "π"), ("rho", "ρ"),
    ("sigma", "σ"), ("tau", "τ"), ("phi", "φ"), ("omega", "ω"), ("Delta", "Δ"),
    ("Sigma", "Σ"), ("Omega", "Ω"), ("Pi", "Π"), ("ldots", "…"), ("dots", "…"),
    ("cdots", "⋯"), ("%", "%"), ("$", "$"), ("{", "{"), ("}", "}"), ("_", "_"),
    ("&", "&"), (",", " "), (";", " "), (":", " "), ("!", ""), ("quad", "  "), ("qquad", "    "),
    ("left", ""), ("right", ""), ("displaystyle", ""), ("\\", "\n"),
];
const SUPERSCRIPT: &[(char, char)] = &[
    ('0', '⁰'), ('1', '¹'), ('2', '²'), ('3', '³'), ('4', '⁴'), ('5', '⁵'), ('6', '⁶'),
    ('7', '⁷'), ('8', '⁸'), ('9', '⁹'), ('+', '⁺'), ('-', '⁻'), ('=', '⁼'), ('(', '⁽'),
    (')', '⁾'), ('n', 'ⁿ'), ('i', 'ⁱ'),
];
const SUBSCRIPT: &[(char, char)] = &[
    ('0', '₀'), ('1', '₁'), ('2', '₂'), ('3', '₃'), ('4', '₄'), ('5', '₅'), ('6', '₆'),
    ('7', '₇'), ('8', '₈'), ('9', '₉'), ('+', '₊'), ('-', '₋'), ('=', '₌'), ('(', '₍'), (')', '₎'),
];

/// Convert one math expression (delimiters already removed).
fn convert_math(math: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = math.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                let name: String = if chars.get(i + 1).is_some_and(|c| c.is_ascii_alphabetic()) {
                    chars[i + 1..].iter().take_while(|c| c.is_ascii_alphabetic()).collect()
                } else {
                    chars.get(i + 1).map(|c| c.to_string()).unwrap_or_default()
                };
                i += 1 + name.chars().count();
                match name.as_str() {
                    "frac" | "dfrac" | "tfrac" => {
                        let (num, next) = group(&chars, i);
                        let (den, next) = group(&chars, next);
                        i = next;
                        out.push_str(&format!("{}/{}", wrap(&convert_math(&num)), wrap(&convert_math(&den))));
                    }
                    "sqrt" => {
                        let (arg, next) = group(&chars, i);
                        i = next;
                        out.push_str(&format!("√{}", wrap(&convert_math(&arg))));
                    }
                    "text" | "textbf" | "mathrm" | "mathbf" | "operatorname" | "boxed" | "mbox" => {
                        let (arg, next) = group(&chars, i);
                        i = next;
                        out.push_str(&convert_math(&arg));
                    }
                    _ => match SYMBOLS.iter().find(|(n, _)| *n == name) {
                        Some((_, symbol)) => out.push_str(symbol),
                        None => out.push_str(&name),
                    },
                }
            }
            c @ ('^' | '_') => {
                let (arg, next) = group(&chars, i + 1);
                i = next;
                let arg = convert_math(&arg);
                let table = if c == '^' { SUPERSCRIPT } else { SUBSCRIPT };
                let mapped: Option<String> =
                    arg.chars().map(|ch| table.iter().find(|(from, _)| *from == ch).map(|(_, to)| *to)).collect();
                match mapped {
                    Some(mapped) => out.push_str(&mapped),
                    None => out.push_str(&format!("{c}({arg})")),
                }
            }
            '{' | '}' => i += 1,
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    out.trim().to_string()
}

/// Read a `{...}` group (or a single character) starting at `i`.
fn group(chars: &[char], mut i: usize) -> (String, usize) {
    while chars.get(i) == Some(&' ') {
        i += 1;
    }
    match chars.get(i) {
        Some('{') => {
            let mut depth = 0;
            let start = i + 1;
            while i < chars.len() {
                match chars[i] {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return (chars[start..i].iter().collect(), i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            (chars[start..].iter().collect(), chars.len())
        }
        Some(&c) => (c.to_string(), i + 1),
        None => (String::new(), i),
    }
}

/// Parenthesize multi-term operands of a fraction or root.
fn wrap(s: &str) -> String {
    if s.chars().all(|c| c.is_alphanumeric() || c == '.') { s.to_string() } else { format!("({s})") }
}

#[cfg(test)]
mod tests {
    use super::latex_to_unicode as f;

    #[test]
    fn inline_and_display_math() {
        assert_eq!(f(r"So \( 12 \times 13 = 156 \)."), "So 12 × 13 = 156.");
        assert_eq!(f(r"\[ x^2 + y_1 \leq 10 \]"), "x² + y₁ ≤ 10");
        assert_eq!(f(r"$$\frac{1}{2} + \frac{a+b}{c}$$"), "1/2 + (a+b)/c");
        assert_eq!(f(r"$\sqrt{16} = 4$"), "√16 = 4");
        assert_eq!(f(r"\( \text{Area} = \pi r^2 \)"), "Area = π r²");
        assert_eq!(f(r"\( 25\% \) of \( 240 \) is \( \boxed{60} \)"), "25% of 240 is 60");
    }

    #[test]
    fn leaves_plain_text_and_money_alone() {
        assert_eq!(f("No math here."), "No math here.");
        assert_eq!(f("It costs $5 and $7 total."), "It costs $5 and $7 total.");
        assert_eq!(f(r"Unclosed \( math"), r"Unclosed \( math");
    }

    #[test]
    fn unknown_superscript_falls_back() {
        assert_eq!(f(r"$e^{x+y}$"), "e^(x+y)");
    }
}
