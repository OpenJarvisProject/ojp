//! Validated Hugging Face repo id newtype.
//!
//! Locked validation shape (per the plan's `Metis` review):
//! `^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?$` — implemented as a per-char loop
//! to avoid pulling in a regex crate.
//!
//! The error variant is `LlamaError::ModelLoad` for now because
//! `LlamaError::ModelDownload` is added in parallel by Task 2 and is not
//! guaranteed to exist when this file is compiled. The resolver (Task 11)
//! may re-map to `ModelDownload` at the integration boundary.

use crate::error::LlamaError;

/// Maximum number of characters allowed in a HF repo id.
const MAX_REPO_ID_LEN: usize = 128;

/// First-segment words that look like local paths and would cause
/// ambiguous dispatch. Reserved in the plan's "non-regression for local
/// paths" guardrail.
const FORBIDDEN_FIRST_SEGMENT: &[&str] = &["models", "model"];

/// A validated Hugging Face repo id (`"org/name"` or bare `"name"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HfRepo(String);

impl HfRepo {
    /// Validate and wrap a string as a `HfRepo`.
    ///
    /// # Errors
    /// Returns `LlamaError::ModelLoad` if `s` is not a valid HF repo id
    /// per the locked shape.
    pub fn new(s: &str) -> Result<Self, LlamaError> {
        Self::validate(s)?;
        Ok(Self(s.to_owned()))
    }

    /// Cheap, side-effect-free check used by the auto-detect resolver.
    /// Mirrors the validation done by [`HfRepo::new`] but returns `bool`.
    #[must_use]
    pub fn looks_like_repo_id(s: &str) -> bool {
        Self::validate(s).is_ok()
    }

    /// Borrow the inner string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Borrow the validated repo id (`"org/name"` or bare `"name"`).
    ///
    /// Canonical name used by the integration layer (resolver,
    /// `Llama::load`, downloader cache keys). Functionally a synonym
    /// for [`HfRepo::as_str`].
    #[must_use]
    pub fn repo_id(&self) -> &str {
        &self.0
    }

    fn validate(s: &str) -> Result<(), LlamaError> {
        if s.is_empty() {
            return Err(invalid("repo id is empty"));
        }
        if s.len() > MAX_REPO_ID_LEN {
            return Err(invalid("repo id exceeds 128 characters"));
        }

        // Hand-rolled: ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?$
        // We split on '/' ourselves; we also forbid an empty segment,
        // a leading slash, a trailing slash, and a double slash.
        let mut iter = s.split('/');
        let first = iter.next().unwrap_or("");

        if first.is_empty() {
            return Err(invalid("repo id has a leading or empty segment"));
        }
        if !is_valid_segment(first) {
            return Err(invalid("repo id segment contains invalid characters"));
        }
        if is_forbidden_first_segment(first) {
            return Err(invalid(
                "repo id first segment is a reserved local-path word",
            ));
        }

        match iter.next() {
            None => Ok(()),
            Some(second) => {
                if second.is_empty() {
                    return Err(invalid("repo id has a trailing or double slash"));
                }
                if !is_valid_segment(second) {
                    return Err(invalid("repo id segment contains invalid characters"));
                }
                if iter.next().is_some() {
                    return Err(invalid("repo id has more than two segments"));
                }
                Ok(())
            }
        }
    }
}

fn invalid(msg: &str) -> LlamaError {
    LlamaError::ModelLoad(format!("invalid HF repo id: {msg}"))
}

fn is_valid_segment(seg: &str) -> bool {
    seg.bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
}

fn is_forbidden_first_segment(seg: &str) -> bool {
    FORBIDDEN_FIRST_SEGMENT
        .iter()
        .any(|bad| seg.eq_ignore_ascii_case(bad))
}

#[cfg(test)]
mod tests {
    use super::HfRepo;

    // ----- accept -----

    #[test]
    fn accepts_two_segment() {
        assert!(HfRepo::new("TheBloke/Llama-2-7B-Chat-GGUF").is_ok());
    }

    #[test]
    fn accepts_single_segment() {
        assert!(HfRepo::new("gpt2").is_ok());
    }

    #[test]
    fn accepts_dots_and_underscores_and_dashes() {
        assert!(HfRepo::new("user.name_with-dots/and-dashes_etc").is_ok());
    }

    // ----- reject -----

    #[test]
    fn rejects_empty() {
        assert!(HfRepo::new("").is_err());
    }

    #[test]
    fn rejects_trailing_slash() {
        assert!(HfRepo::new("TheBloke/").is_err());
    }

    #[test]
    fn rejects_leading_slash() {
        assert!(HfRepo::new("/TheBloke/Llama").is_err());
    }

    #[test]
    fn rejects_relative_prefix() {
        // "./TheBloke/Llama" is 3 segments: ".", "TheBloke", "Llama".
        assert!(HfRepo::new("./TheBloke/Llama").is_err());
    }

    #[test]
    fn rejects_tilde() {
        // '~' is not in [A-Za-z0-9._-].
        assert!(HfRepo::new("~/models/foo").is_err());
    }

    #[test]
    fn rejects_drive_letter() {
        // ':' is not in [A-Za-z0-9._-].
        assert!(HfRepo::new("C:/foo/bar").is_err());
    }

    #[test]
    fn rejects_url() {
        // ':' and '/' beyond a single separator are both rejected.
        assert!(HfRepo::new("https://huggingface.co/TheBloke/Llama").is_err());
    }

    #[test]
    fn rejects_no_slash() {
        // "models" is a reserved local-path word; even with a slash it
        // must be rejected so the resolver can fall through to Local.
        assert!(HfRepo::new("models/foo").is_err());
    }

    #[test]
    fn rejects_double_slash() {
        // Second segment is empty -> trailing/double slash.
        assert!(HfRepo::new("TheBloke//Llama").is_err());
    }

    #[test]
    fn rejects_three_segments() {
        // A file name must come from the builder, not the repo id.
        assert!(HfRepo::new("TheBloke/Llama/file.gguf").is_err());
    }

    // ----- looks_like_repo_id mirrors new -----

    #[test]
    fn looks_like_repo_id_matches_new() {
        for ok in [
            "gpt2",
            "TheBloke/Llama-2-7B-Chat-GGUF",
            "user.name/repo_1-2",
        ] {
            assert_eq!(
                HfRepo::looks_like_repo_id(ok),
                HfRepo::new(ok).is_ok(),
                "input: {ok}"
            );
        }
        for bad in [
            "",
            "/x",
            "x/",
            "a//b",
            "a/b/c",
            "https://huggingface.co/x/y",
        ] {
            assert!(!HfRepo::looks_like_repo_id(bad), "input: {bad}");
        }
    }

    #[test]
    fn returns_inner_string_unchanged() {
        let repo = HfRepo::new("TheBloke/Foo").expect("valid");
        assert_eq!(repo.repo_id(), "TheBloke/Foo");
    }
}
