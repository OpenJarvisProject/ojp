//! The "hands" beyond simple desktop controls: files, the web and a terminal.
//! Everything here is built to be safe to hand to a small model that can be
//! fooled by text it reads (a web page or a file can contain instructions):
//!
//! - Reading is limited to your home folder, with secrets (keys, browser
//!   profiles, password stores, .env files) always off limits.
//! - Anything that changes something (writing a file, running a command)
//!   is never done directly: the engine asks you first (see `Risk`).
//! - The web is off until you turn it on in the config (`[tools] internet`),
//!   and even then pages on your own network or machine are refused.
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

/// Switches the user controls (config `[tools]`).
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    /// Allow web_search and web_fetch.
    pub internet: bool,
}

static POLICY: RwLock<Policy> = RwLock::new(Policy { internet: false });

pub fn set_policy(policy: Policy) {
    *POLICY.write().unwrap_or_else(|e| e.into_inner()) = policy;
}

fn policy() -> Policy {
    *POLICY.read().unwrap_or_else(|e| e.into_inner())
}

fn home() -> Result<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from).filter(|h| h.is_absolute()).context("no home folder")
}

/// Folders and files that are never read or written, relative to home.
const SECRET_PATHS: &[&str] = &[
    ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".password-store", ".local/share/keyrings",
    ".mozilla", ".config/BraveSoftware", ".config/google-chrome", ".config/chromium", ".config/Code",
    ".config/ojp", ".local/share/ojp/chats", ".claude", ".netrc", ".git-credentials", ".npmrc", ".pypirc",
];

fn is_secret(path: &Path, home: &Path) -> bool {
    let Ok(rel) = path.strip_prefix(home) else { return true };
    if SECRET_PATHS.iter().any(|s| rel.starts_with(s)) {
        return true;
    }
    let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    name == ".env"
        || name.starts_with(".env.")
        || name.ends_with(".pem")
        || name.ends_with(".key")
        || name.starts_with("id_rsa")
        || name.starts_with("id_ed25519")
        || name.contains("credentials")
        || name.contains("password")
}

/// Resolve a user-supplied path ("~/Documents/a.txt", "Documents/a.txt") to a
/// real path inside the home folder that is not a secret.
pub fn resolve(input: &str) -> Result<PathBuf> {
    let home = home()?.canonicalize().context("home folder")?;
    let input = input.trim();
    let raw = if let Some(rest) = input.strip_prefix("~/") {
        home.join(rest)
    } else if input == "~" {
        home.clone()
    } else if Path::new(input).is_absolute() {
        PathBuf::from(input)
    } else {
        home.join(input)
    };
    // Resolve symlinks and ".." on the part that exists, so a link can't lead out.
    let mut existing = raw.clone();
    let mut tail = Vec::new();
    while !existing.exists() {
        match existing.file_name() {
            Some(n) => tail.push(n.to_os_string()),
            None => bail!("{input:?} isn't a valid path"),
        }
        existing.pop();
    }
    let mut resolved = existing.canonicalize()?;
    for part in tail.into_iter().rev() {
        if part == ".." {
            bail!("{input:?} isn't a valid path");
        }
        resolved.push(part);
    }
    if !resolved.starts_with(&home) {
        bail!("{input:?} is outside your home folder, which I don't touch.");
    }
    if is_secret(&resolved, &home) {
        bail!("{input:?} looks like a private file or folder (keys, passwords, browser data), so I won't open it.");
    }
    Ok(resolved)
}

pub fn shorten(path: &Path) -> String {
    match home().ok().and_then(|h| path.strip_prefix(h).ok().map(|p| p.to_path_buf())) {
        Some(rel) => format!("~/{}", rel.display()),
        None => path.display().to_string(),
    }
}

/// First `max` characters of `text`, with a note if cut.
fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    format!("{}… (cut off, {} characters in all)", text.chars().take(max).collect::<String>(), text.chars().count())
}

pub fn read_file(path: &str) -> Result<String> {
    let path = resolve(path)?;
    let meta = std::fs::metadata(&path).with_context(|| format!("{} doesn't exist", shorten(&path)))?;
    if meta.is_dir() {
        bail!("{} is a folder; list it instead", shorten(&path));
    }
    if meta.len() > 2_000_000 {
        bail!("{} is too big to read ({} MB)", shorten(&path), meta.len() / 1_000_000);
    }
    let bytes = std::fs::read(&path)?;
    if bytes.contains(&0) {
        bail!("{} isn't a text file", shorten(&path));
    }
    let text = String::from_utf8_lossy(&bytes);
    Ok(format!("{}:\n{}", shorten(&path), clip(text.trim(), 3000)))
}

pub fn list_dir(path: &str) -> Result<String> {
    let path = resolve(if path.trim().is_empty() { "~" } else { path })?;
    let read = std::fs::read_dir(&path).with_context(|| format!("{} isn't a folder I can open", shorten(&path)))?;
    let home = home()?.canonicalize()?;
    let mut entries: Vec<(bool, String)> = read
        .flatten()
        .filter(|e| !e.file_name().to_string_lossy().starts_with('.') && !is_secret(&e.path(), &home))
        .map(|e| (e.path().is_dir(), e.file_name().to_string_lossy().into_owned()))
        .collect();
    entries.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.to_lowercase().cmp(&b.1.to_lowercase())));
    if entries.is_empty() {
        return Ok(format!("{} is empty.", shorten(&path)));
    }
    let total = entries.len();
    let shown: Vec<String> = entries.iter().take(40).map(|(dir, n)| if *dir { format!("{n}/") } else { n.clone() }).collect();
    Ok(format!("{} ({total} items{}):\n{}", shorten(&path), if total > 40 { ", first 40" } else { "" }, shown.join("\n")))
}

/// Search inside text files in the home folder for `query` (case-insensitive);
/// returns up to 8 matching lines with their file names.
pub fn search_text(query: &str) -> Result<String> {
    let query = query.trim().to_lowercase();
    if query.chars().count() < 2 {
        bail!("give me at least two letters to search for");
    }
    // Lines are ranked by how many of the words they contain, so "rent amount"
    // still finds the line "Rent 1200".
    let words: Vec<&str> = query.split_whitespace().filter(|w| w.chars().count() >= 3 || query.split_whitespace().count() == 1).collect();
    let words = if words.is_empty() { vec![query.as_str()] } else { words };
    let home = home()?.canonicalize()?;
    let started = Instant::now();
    let mut hits: Vec<(usize, String)> = Vec::new();
    let mut files_seen = 0usize;
    let mut stack = vec![(home.clone(), 0usize)];
    'walk: while let Some((dir, depth)) = stack.pop() {
        let Ok(read) = std::fs::read_dir(&dir) else { continue };
        let mut entries: Vec<_> = read.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || is_secret(&path, &home) || matches!(name.as_str(), "node_modules" | "target" | "__pycache__" | "venv" | "SteamLibrary") {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if meta.file_type().is_symlink() {
                continue;
            }
            if meta.is_dir() {
                if depth < 6 {
                    stack.push((path, depth + 1));
                }
            } else if meta.len() <= 1_000_000 {
                files_seen += 1;
                if files_seen > 20_000 || started.elapsed() > Duration::from_secs(8) {
                    break 'walk;
                }
                let Ok(bytes) = std::fs::read(&path) else { continue };
                if bytes.contains(&0) {
                    continue;
                }
                let text = String::from_utf8_lossy(&bytes);
                // The best-matching line of each file.
                let best = text
                    .lines()
                    .enumerate()
                    .map(|(n, line)| {
                        let l = line.to_lowercase();
                        (words.iter().filter(|w| l.contains(**w)).count(), n, line)
                    })
                    .filter(|(score, _, _)| *score > 0)
                    .max_by_key(|(score, n, _)| (*score, std::cmp::Reverse(*n)));
                if let Some((score, n, line)) = best {
                    hits.push((score, format!("{}:{}: {}", shorten(&path), n + 1, clip(line.trim(), 160))));
                    if hits.len() >= 40 {
                        break 'walk;
                    }
                }
            }
        }
    }
    if hits.is_empty() {
        return Ok(format!("No text file in your home folder mentions {query:?}."));
    }
    hits.sort_by(|a, b| b.0.cmp(&a.0));
    let lines: Vec<String> = hits.into_iter().take(8).map(|(_, h)| h).collect();
    Ok(format!("Found in {} file(s):\n{}", lines.len(), lines.join("\n")))
}

/// Describe what writing would do (shown to the user before they agree).
pub fn write_plan(path: &str, text: &str) -> Result<String> {
    let resolved = resolve(path)?;
    let hidden = resolved.strip_prefix(home()?.canonicalize()?).ok().is_some_and(|r| {
        r.components().any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
    });
    if hidden {
        bail!("{path:?} is a hidden file or folder (settings), so I won't write there.");
    }
    let what = if resolved.exists() { "replace the existing file" } else { "create a new file" };
    Ok(format!("write {} characters to {} ({what})", text.chars().count(), shorten(&resolved)))
}

pub fn write_file(path: &str, text: &str) -> Result<String> {
    write_plan(path, text)?; // the same checks, again, right before writing
    let resolved = resolve(path)?;
    if let Some(dir) = resolved.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&resolved, text)?;
    Ok(format!("Wrote {}.", shorten(&resolved)))
}

/// Run a shell command in the home folder (only ever after the user agreed).
pub fn run_command(command: &str) -> Result<String> {
    let command = command.trim();
    if command.is_empty() {
        bail!("no command to run");
    }
    let mut child = std::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(home()?)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("couldn't start the command")?;
    let start = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if start.elapsed() > Duration::from_secs(30) {
            let _ = child.kill();
            let _ = child.wait();
            bail!("the command ran for over 30 seconds, so I stopped it");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let out = child.wait_with_output()?;
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    let err = String::from_utf8_lossy(&out.stderr);
    if !err.trim().is_empty() {
        text.push_str(&format!("\n[errors] {}", err.trim()));
    }
    let code = out.status.code().map_or("signal".to_string(), |c| c.to_string());
    Ok(format!("exit code {code}\n{}", clip(text.trim(), 2500)))
}

// --- the web ---------------------------------------------------------------

fn require_internet() -> Result<()> {
    if !policy().internet {
        bail!("internet access is off. Turn it on with `internet = true` under [tools] in your config if you want me to use the web.");
    }
    Ok(())
}

/// Refuse addresses that point at this machine or your own network.
pub fn check_public_url(url: &str) -> Result<()> {
    let rest = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://")).context("only http(s) addresses")?;
    let host_port = rest.split(['/', '?', '#']).next().unwrap_or("");
    if host_port.contains('@') {
        bail!("addresses with a login in them aren't allowed");
    }
    let host = host_port.rsplit_once(':').map_or(host_port, |(h, p)| if p.chars().all(|c| c.is_ascii_digit()) { h } else { host_port });
    let host = host.trim_matches(['[', ']']).to_lowercase();
    if host.is_empty() {
        bail!("no host in that address");
    }
    let private_name = host == "localhost" || host.ends_with(".local") || host.ends_with(".internal") || host.ends_with(".lan") || !host.contains('.') && !host.contains(':');
    let private_ip = host.parse::<std::net::IpAddr>().is_ok_and(|ip| match ip {
        std::net::IpAddr::V4(v) => v.is_private() || v.is_loopback() || v.is_link_local() || v.is_unspecified() || v.octets()[0] == 100 && (64..128).contains(&v.octets()[1]),
        std::net::IpAddr::V6(v) => v.is_loopback() || v.is_unspecified() || (v.segments()[0] & 0xfe00) == 0xfc00 || (v.segments()[0] & 0xffc0) == 0xfe80,
    });
    if private_name || private_ip {
        bail!("{host} is on this computer or your own network, so I won't fetch it");
    }
    Ok(())
}

fn http_get(url: &str) -> Result<String> {
    check_public_url(url)?;
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(12))).build().into();
    let mut response = agent.get(url).header("User-Agent", "OpenJarvis/0.1 (+https://github.com/OpenJarvisProject/ojp)").call().context("couldn't reach that page")?;
    let body = response.body_mut().with_config().limit(1_500_000).read_to_string().context("couldn't read that page")?;
    Ok(body)
}

pub fn web_fetch(url: &str) -> Result<String> {
    require_internet()?;
    let body = http_get(url.trim())?;
    let text = html_to_text(&body);
    if text.is_empty() {
        bail!("that page has no readable text");
    }
    Ok(format!("{url}:\n{}", clip(&text, 3500)))
}

pub fn web_search(query: &str) -> Result<String> {
    require_internet()?;
    let url = format!("https://html.duckduckgo.com/html/?q={}", percent_encode(query.trim()));
    let body = http_get(&url)?;
    let results = parse_search_results(&body);
    if results.is_empty() {
        return Ok(format!("No web results for {query:?}."));
    }
    let lines: Vec<String> = results.iter().take(5).enumerate().map(|(i, r)| format!("{}. {}\n   {}\n   {}", i + 1, r.0, r.2, r.1)).collect();
    Ok(format!("Web results for {query:?}:\n{}", lines.join("\n")))
}

fn percent_encode(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// (title, url, snippet) from DuckDuckGo's plain-HTML result page.
pub fn parse_search_results(html: &str) -> Vec<(String, String, String)> {
    let mut results = Vec::new();
    let mut rest = html;
    while let Some(i) = rest.find("class=\"result__a\"") {
        rest = &rest[i..];
        let href = attr(rest, "href").unwrap_or_default();
        let title = rest.find('>').map(|s| html_to_text(rest[s + 1..].split("</a>").next().unwrap_or(""))).unwrap_or_default();
        let url = match href.split_once("uddg=") {
            Some((_, enc)) => percent_decode(enc.split('&').next().unwrap_or("")),
            None => href.clone(),
        };
        let next = rest[1..].find("class=\"result__a\"").map_or(rest.len(), |n| n + 1);
        let snippet = rest[..next]
            .find("class=\"result__snippet\"")
            .and_then(|s| rest[s..].find('>').map(|g| html_to_text(rest[s + g + 1..].split("</a>").next().unwrap_or(""))))
            .unwrap_or_default();
        if !title.is_empty() && url.starts_with("http") {
            results.push((title, url, snippet));
        }
        rest = &rest[next.min(rest.len())..];
        if rest.is_empty() {
            break;
        }
    }
    results
}

fn attr(tag_start: &str, name: &str) -> Option<String> {
    let end = tag_start.find('>')?;
    let tag = &tag_start[..end];
    let key = format!("{name}=\"");
    let i = tag.find(&key)? + key.len();
    Some(tag[i..].split('"').next()?.to_string())
}

/// Readable text from HTML: drops scripts, styles and tags, decodes common entities.
pub fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 3);
    let lower = html.to_lowercase();
    let (mut i, bytes) = (0, html.as_bytes());
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip whole script/style blocks.
            let rest = &lower[i..];
            let block = ["<script", "<style", "<noscript", "<svg", "<head"].iter().find(|t| rest.starts_with(**t));
            if let Some(tag) = block {
                let close = format!("</{}", &tag[1..]);
                i = lower[i..].find(&close).map_or(bytes.len(), |c| i + c);
            }
            let end = html[i..].find('>').map_or(bytes.len(), |e| i + e + 1);
            let name = lower[i..end.min(lower.len())].trim_start_matches(['<', '/']).split([' ', '>', '/']).next().unwrap_or("");
            if matches!(name, "p" | "br" | "div" | "li" | "tr" | "h1" | "h2" | "h3" | "h4" | "section" | "article") {
                out.push('\n');
            }
            i = end;
        } else {
            let next = html[i..].find('<').map_or(bytes.len(), |n| i + n);
            out.push_str(&html[i..next]);
            i = next;
        }
    }
    let decoded = out
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&rsquo;", "'")
        .replace("&mdash;", "—");
    let lines: Vec<String> = decoded.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|l| !l.is_empty()).collect();
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::TEST_ENV as ENV;

    fn fake_home() -> (std::sync::MutexGuard<'static, ()>, PathBuf) {
        let guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("ojp-home-{}-{}", std::process::id(), Instant::now().elapsed().as_nanos()));
        std::fs::create_dir_all(dir.join("Documents")).unwrap();
        std::fs::create_dir_all(dir.join(".ssh")).unwrap();
        std::fs::write(dir.join("Documents/resume.txt"), "Jane Doe, engineer").unwrap();
        std::fs::write(dir.join(".ssh/id_rsa"), "SECRET").unwrap();
        std::fs::write(dir.join("Documents/.env"), "TOKEN=1").unwrap();
        unsafe { std::env::set_var("HOME", &dir) };
        (guard, dir.canonicalize().unwrap())
    }

    #[test]
    fn reads_inside_home_and_refuses_secrets_and_outside() {
        let (_g, home) = fake_home();
        assert!(read_file("Documents/resume.txt").unwrap().contains("Jane Doe"));
        assert!(read_file("~/Documents/resume.txt").unwrap().contains("Jane Doe"));
        assert!(read_file(&format!("{}/Documents/resume.txt", home.display())).is_ok());
        assert!(read_file(".ssh/id_rsa").unwrap_err().to_string().contains("private"));
        assert!(read_file("Documents/.env").is_err());
        assert!(read_file("/etc/passwd").unwrap_err().to_string().contains("outside"));
        assert!(read_file("../../etc/passwd").is_err());
        assert!(read_file("Documents").unwrap_err().to_string().contains("folder"));
        std::fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn symlinks_cannot_lead_out_of_home() {
        let (_g, home) = fake_home();
        std::os::unix::fs::symlink("/etc", home.join("etc-link")).unwrap();
        assert!(read_file("etc-link/passwd").unwrap_err().to_string().contains("outside"));
        std::fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn lists_folders_without_hidden_or_secret_entries() {
        let (_g, home) = fake_home();
        let top = list_dir("").unwrap();
        assert!(top.contains("Documents/") && !top.contains(".ssh"), "{top}");
        let docs = list_dir("Documents").unwrap();
        assert!(docs.contains("resume.txt") && !docs.contains(".env"), "{docs}");
        std::fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn searches_inside_files_but_not_secrets() {
        let (_g, home) = fake_home();
        std::fs::write(home.join("Documents/budget.txt"), "Rent 1200\nFood 400").unwrap();
        std::fs::write(home.join(".ssh/config"), "rent secret").unwrap();
        let out = search_text("rent").unwrap();
        assert!(out.contains("Documents/budget.txt:1: Rent 1200"), "{out}");
        assert!(!out.contains(".ssh"), "{out}");
        assert!(search_text("zzzqqq").unwrap().starts_with("No text file"));
        // Several words: the line with the most of them comes first.
        std::fs::write(home.join("Documents/other.txt"), "the amount of rent due is on the first").unwrap();
        let out = search_text("rent amount").unwrap();
        assert!(out.lines().nth(1).unwrap().contains("other.txt"), "{out}");
        assert!(out.contains("budget.txt"), "{out}");
        assert!(search_text("x").is_err());
        std::fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn writing_is_planned_checked_and_limited() {
        let (_g, home) = fake_home();
        assert!(write_plan("Documents/new.txt", "hi").unwrap().contains("create a new file"));
        assert!(write_plan("Documents/resume.txt", "hi").unwrap().contains("replace"));
        assert!(write_plan(".bashrc", "x").is_err());
        assert!(write_plan(".config/autostart/x.desktop", "x").is_err());
        assert!(write_plan("/etc/hosts", "x").is_err());
        assert_eq!(write_file("Documents/notes/new.txt", "hello").unwrap(), "Wrote ~/Documents/notes/new.txt.");
        assert_eq!(std::fs::read_to_string(home.join("Documents/notes/new.txt")).unwrap(), "hello");
        std::fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn commands_run_with_a_time_limit_and_report_output() {
        let (_g, home) = fake_home();
        let out = run_command("echo hi; echo oops 1>&2; exit 3").unwrap();
        assert!(out.starts_with("exit code 3") && out.contains("hi") && out.contains("oops"), "{out}");
        assert!(run_command("").is_err());
        std::fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn web_is_off_by_default_and_refuses_local_addresses() {
        set_policy(Policy { internet: false });
        assert!(web_fetch("https://example.com").unwrap_err().to_string().contains("internet access is off"));
        for bad in ["http://localhost:8080/x", "http://127.0.0.1/", "http://192.168.1.1/", "http://10.0.0.5/a", "http://169.254.169.254/latest", "http://[::1]/", "http://printer.local/", "http://intranet/", "https://user:pw@example.com/", "ftp://example.com/", "http://100.64.0.1/"] {
            assert!(check_public_url(bad).is_err(), "{bad} should be refused");
        }
        for good in ["https://example.com/a?b=c", "http://en.wikipedia.org/wiki/Rust", "https://1.1.1.1/"] {
            assert!(check_public_url(good).is_ok(), "{good} should be allowed");
        }
    }

    #[test]
    fn html_becomes_text() {
        let html = "<html><head><title>x</title><style>p{}</style></head><body><h1>Hi &amp; welcome</h1><script>evil()</script><p>First  line</p><p>Second <b>bold</b> it&#39;s</p></body></html>";
        assert_eq!(html_to_text(html), "Hi & welcome\nFirst line\nSecond bold it's");
    }

    #[test]
    fn search_results_are_parsed() {
        let html = r#"<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rust-lang.org%2F&amp;rut=abc">Rust <b>Programming</b> Language</a><a class="result__snippet" href="x">A language empowering everyone.</a></div><div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdoc.rust-lang.org%2Fbook%2F&amp;rut=def">The Book</a><a class="result__snippet" href="y">Learn Rust.</a></div>"#;
        let r = parse_search_results(html);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0], ("Rust Programming Language".into(), "https://www.rust-lang.org/".into(), "A language empowering everyone.".into()));
        assert_eq!(r[1].1, "https://doc.rust-lang.org/book/");
    }
}
