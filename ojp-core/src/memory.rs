//! Long-term memory: things the assistant knows about the user, kept in a
//! private file on the device (`memory.jsonl`, one JSON object per line, readable
//! only by the user). Nothing is sent anywhere.
//!
//! Each memory has its full text, a short summary (what the model reads when
//! it only needs the gist, which saves tokens) and a few tags from a small
//! fixed vocabulary ("cooking", "schedule", "people", ...). Words that mean
//! nearly the same thing ("baking", "recipe", "dinner") all map to one tag
//! ("cooking"), so there are never endless near-duplicate tags; only the user
//! adds tags (`/tag add`).
//!
//! Recall is keyword search ranked with BM25, plus a rule that keeps it useful
//! while the memory is small: with up to 15 memories, all of them are given to
//! the model, so a question phrased with different words still finds the
//! answer. (Search by meaning, with embeddings, can be added behind the same
//! interface later.)
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Memory {
    pub id: u64,
    /// Seconds since 1970 when it was saved.
    pub created: u64,
    pub text: String,
    /// A short version (at most about 90 characters). Empty in older files: use the text.
    #[serde(default)]
    pub summary: String,
    /// Tags from the vocabulary, lowercase.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl Memory {
    /// What to show when the full text isn't needed.
    pub fn gist(&self) -> &str {
        if self.summary.is_empty() { &self.text } else { &self.summary }
    }
}

/// The built-in vocabulary: tag, then the words that mean it. Matching ignores
/// case and a trailing "s".
const SEED_TAGS: &[(&str, &[&str])] = &[
    ("people", &["family", "friend", "sister", "brother", "mother", "mom", "mum", "father", "dad", "wife", "husband", "partner", "girlfriend", "boyfriend", "kid", "son", "daughter", "cousin", "aunt", "uncle", "grandma", "grandpa", "colleague", "coworker", "boss", "neighbor", "neighbour"]),
    ("cooking", &["baking", "bake", "recipe", "food", "meal", "dinner", "lunch", "breakfast", "kitchen", "diet", "vegetarian", "vegan", "cuisine", "snack", "tea", "coffee", "drink", "ingredient", "restaurant"]),
    ("schedule", &["calendar", "appointment", "meeting", "shift", "deadline", "plan", "agenda", "timetable", "weekday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "birthday", "event"]),
    ("work", &["job", "career", "office", "project", "company", "employer", "salary", "client", "manager", "workplace"]),
    ("health", &["allergy", "allergic", "medication", "medicine", "doctor", "dentist", "exercise", "fitness", "sleep", "injury", "therapy", "hospital", "symptom"]),
    ("hobbies", &["hobby", "music", "game", "gaming", "reading", "book", "sport", "movie", "film", "show", "art", "drawing", "photography", "guitar", "piano", "anime"]),
    ("preferences", &["favorite", "favourite", "like", "love", "dislike", "hate", "prefer", "enjoy", "fan"]),
    ("places", &["address", "city", "travel", "trip", "vacation", "holiday", "flight", "hotel", "country", "town", "street", "moved"]),
    ("tech", &["computer", "laptop", "phone", "software", "linux", "code", "coding", "programming", "server", "network", "app", "keyboard", "android"]),
    ("money", &["budget", "bank", "finance", "bill", "rent", "saving", "debt", "invoice", "tax", "salary", "price", "paid"]),
    ("pets", &["pet", "dog", "cat", "puppy", "kitten", "hamster", "fish"]),
    ("home", &["house", "apartment", "flat", "garden", "furniture", "cleaning", "laundry", "room", "roommate"]),
    ("learning", &["study", "course", "language", "russian", "class", "exam", "school", "university", "learn", "homework"]),
];

/// The tag used when nothing in the vocabulary fits.
pub const OTHER: &str = "other";

fn tags_path() -> PathBuf {
    path().with_file_name("memory-tags.json")
}

/// The vocabulary: built-in tags plus whatever the user added or extended.
pub fn vocabulary() -> Vec<(String, Vec<String>)> {
    let mut vocab: Vec<(String, Vec<String>)> =
        SEED_TAGS.iter().map(|(t, w)| (t.to_string(), w.iter().map(|x| x.to_string()).collect())).collect();
    let user: HashMap<String, Vec<String>> =
        std::fs::read_to_string(tags_path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    for (tag, words) in user {
        match vocab.iter_mut().find(|(t, _)| *t == tag) {
            Some((_, existing)) => existing.extend(words),
            None => vocab.push((tag, words)),
        }
    }
    vocab
}

fn stem(word: &str) -> String {
    let w = word.to_lowercase();
    let w = w.trim_end_matches('s');
    if w.is_empty() { "s".into() } else { w.to_string() }
}

/// Which tag a single word means, if any ("baking" -> "cooking").
pub fn tag_for_word(word: &str) -> Option<String> {
    let w = stem(word);
    vocabulary().into_iter().find(|(tag, words)| stem(tag) == w || words.iter().any(|x| stem(x) == w)).map(|(tag, _)| tag)
}

/// Tags for a text, by its words (at most three; "other" if none fit).
pub fn auto_tags(text: &str) -> Vec<String> {
    let vocab = vocabulary();
    let words: Vec<String> = text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(stem).collect();
    let mut scored: Vec<(usize, String)> = vocab
        .iter()
        .map(|(tag, aliases)| {
            let hits = words.iter().filter(|w| **w == stem(tag) || aliases.iter().any(|a| stem(a) == **w)).count();
            (hits, tag.clone())
        })
        .filter(|(h, _)| *h > 0)
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    let mut tags: Vec<String> = scored.into_iter().take(3).map(|(_, t)| t).collect();
    if tags.is_empty() {
        tags.push(OTHER.into());
    }
    tags
}

/// A short version of a text: the text itself when short, else its first clause.
pub fn auto_summary(text: &str) -> String {
    let t = text.trim();
    if t.chars().count() <= 90 {
        return t.to_string();
    }
    let first = t.split(['.', ';', '!', '?']).next().unwrap_or(t).trim();
    let cut: String = first.chars().take(90).collect();
    if first.chars().count() > 90 { format!("{}…", cut.trim_end()) } else { cut }
}

/// Add a tag (or extra words for an existing tag) to the vocabulary.
pub fn add_vocabulary(tag: &str, words: &[String]) -> Result<()> {
    let tag = tag.trim().to_lowercase();
    if tag.is_empty() || !tag.chars().all(|c| c.is_alphanumeric() || c == '-') || tag.len() > 24 {
        bail!("a tag is one short word");
    }
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = tags_path();
    let mut user: HashMap<String, Vec<String>> =
        std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let entry = user.entry(tag).or_default();
    for w in words {
        let w = w.trim().to_lowercase();
        if !w.is_empty() && !entry.contains(&w) {
            entry.push(w);
        }
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(&user)?)?;
    Ok(())
}

/// Every tag with how many memories carry it, most used first.
pub fn tag_counts() -> Vec<(String, usize)> {
    let memories = list();
    let mut counts: Vec<(String, usize)> = vocabulary()
        .into_iter()
        .map(|(t, _)| (t.clone(), memories.iter().filter(|m| m.tags.contains(&t)).count()))
        .collect();
    counts.push((OTHER.into(), memories.iter().filter(|m| m.tags.iter().any(|t| t == OTHER)).count()));
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    counts
}

/// Replace a memory's tags and summary (used after the model has looked at it).
pub fn set_meta(id: u64, tags: Vec<String>, summary: String) -> Result<()> {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut list = read();
    let m = list.iter_mut().find(|m| m.id == id).with_context(|| format!("no memory #{id}"))?;
    m.tags = tags;
    m.summary = summary;
    write(&list)
}

/// Memories carrying `tag`, newest first.
pub fn with_tag(tag: &str) -> Vec<Memory> {
    let mut found: Vec<Memory> = list().into_iter().filter(|m| m.tags.iter().any(|t| t == tag)).collect();
    found.sort_by(|a, b| b.created.cmp(&a.created));
    found
}

/// Serializes read-modify-write cycles on the file.
static LOCK: Mutex<()> = Mutex::new(());

/// With this many memories or fewer, all of them go to the model.
const SMALL: usize = 15;
/// How many ranked memories go to the model once there are more than `SMALL`.
const TOP: usize = 6;
/// A saved memory is at most this long.
pub const MAX_LEN: usize = 300;

fn path() -> PathBuf {
    if let Some(p) = std::env::var_os("OJP_MEMORY_FILE") {
        return PathBuf::from(p);
    }
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".local/share")).join("ojp/memory.jsonl")
}

fn read() -> Vec<Memory> {
    let text = std::fs::read_to_string(path()).unwrap_or_default();
    text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
}

fn write(list: &[Memory]) -> Result<()> {
    let path = path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut body = String::new();
    for m in list {
        body.push_str(&serde_json::to_string(m)?);
        body.push('\n');
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, body)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(tmp, path).context("saving memory")
}

fn now() -> u64 {
    crate::reminders::now()
}

/// Words that carry no meaning for search.
const STOP: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "i", "me", "my", "mine", "you", "your", "we", "our", "it", "its",
    "of", "to", "in", "on", "at", "for", "with", "and", "or", "but", "do", "does", "did", "have", "has", "had", "that", "this", "what",
    "whats", "who", "which", "when", "where", "how", "about", "tell", "remember", "know", "can", "could", "please", "any", "some",
    "from", "by", "as", "so", "if", "than", "then", "there", "their", "they", "them", "he", "she", "his", "her", "s",
];

/// Lowercase words, without filler words and with plurals trimmed.
fn terms(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !STOP.contains(w))
        .map(|w| {
            let w = w.trim_end_matches('s');
            if w.is_empty() { "s".to_string() } else { w.to_string() }
        })
        .collect()
}

/// A memory cleaned for saving: one line, not too long.
pub fn clean(text: &str) -> Result<String> {
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let t = t.trim().trim_end_matches(['.', '!']).trim();
    if t.chars().count() < 3 {
        bail!("there's nothing to remember in that");
    }
    if t.chars().count() > MAX_LEN {
        bail!("that's too long to remember as one thing (over {MAX_LEN} characters)");
    }
    Ok(t.to_string())
}

/// Save a memory. Saying the same thing again does not add it twice.
pub fn add(text: &str) -> Result<Memory> {
    let text = clean(text)?;
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut list = read();
    let key = terms(&text);
    if let Some(existing) = list.iter().find(|m| m.text.eq_ignore_ascii_case(&text) || (terms(&m.text) == key && !key.is_empty())) {
        return Ok(existing.clone());
    }
    let id = list.iter().map(|m| m.id).max().unwrap_or(0) + 1;
    let (tags, summary) = (auto_tags(&text), auto_summary(&text));
    let memory = Memory { id, created: now(), text, summary, tags };
    list.push(memory.clone());
    write(&list)?;
    Ok(memory)
}

pub fn list() -> Vec<Memory> {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    read()
}

/// BM25 relevance of each memory to `query`, best first (only those that share a word).
pub fn search(query: &str, limit: usize) -> Vec<(f32, Memory)> {
    let memories = list();
    rank(&memories, query).into_iter().take(limit).collect()
}

fn rank(memories: &[Memory], query: &str) -> Vec<(f32, Memory)> {
    let q: HashSet<String> = terms(query).into_iter().collect();
    if q.is_empty() || memories.is_empty() {
        return Vec::new();
    }
    // Tags the question is about ("baking tips" -> cooking): memories with one
    // of them are relevant even when they share no word with the question.
    let query_tags: HashSet<String> = query.split(|c: char| !c.is_alphanumeric()).filter_map(tag_for_word).collect();
    let docs: Vec<Vec<String>> = memories.iter().map(|m| terms(&m.text)).collect();
    let n = docs.len() as f32;
    let avg_len = docs.iter().map(Vec::len).sum::<usize>().max(1) as f32 / n;
    let mut df: HashMap<&str, f32> = HashMap::new();
    for d in &docs {
        for t in d.iter().collect::<HashSet<_>>() {
            *df.entry(t.as_str()).or_default() += 1.0;
        }
    }
    let (k1, b) = (1.4_f32, 0.75_f32);
    let mut scored: Vec<(f32, Memory)> = docs
        .iter()
        .zip(memories)
        .filter_map(|(d, m)| {
            let len = d.len() as f32;
            let score: f32 = q
                .iter()
                .map(|term| {
                    let tf = d.iter().filter(|t| *t == term).count() as f32;
                    if tf == 0.0 {
                        return 0.0;
                    }
                    let dfv = df.get(term.as_str()).copied().unwrap_or(0.0);
                    let idf = ((n - dfv + 0.5) / (dfv + 0.5) + 1.0).ln();
                    idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * len / avg_len))
                })
                .sum();
            let tag_bonus = if m.tags.iter().any(|t| query_tags.contains(t)) { 1.5 } else { 0.0 };
            let score = score + tag_bonus;
            (score > 0.0).then(|| (score, m.clone()))
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.created.cmp(&a.1.created)));
    scored
}

/// What to tell the model about the user for this request, or None if there is nothing.
pub fn context_for(request: &str) -> Option<String> {
    let all = list();
    if all.is_empty() {
        return None;
    }
    let chosen: Vec<Memory> = if all.len() <= SMALL {
        all
    } else {
        let mut picked: Vec<Memory> = rank(&all, request).into_iter().take(TOP).map(|(_, m)| m).collect();
        // The newest few as well: they are the likeliest to matter right now.
        for m in all.iter().rev().take(3) {
            if !picked.iter().any(|p| p.id == m.id) {
                picked.push(m.clone());
            }
        }
        picked
    };
    // Each line is attributed to the user so a small model doesn't take "I love tea" as its own.
    let lines: Vec<String> = chosen.iter().map(|m| format!("- The user told you: \"{}\"", m.gist())).collect();
    Some(format!("[Your memory of the user:\n{}]", lines.join("\n")))
}

/// The memories a forget request would remove: `#3` is memory number 3, otherwise
/// every memory containing the words (all of them), or failing that the single best match.
pub fn forget_matches(query: &str) -> Result<Vec<Memory>> {
    let all = list();
    let query = query.trim();
    if let Some(id) = query.strip_prefix('#').and_then(|n| n.trim().parse::<u64>().ok()) {
        let found: Vec<Memory> = all.into_iter().filter(|m| m.id == id).collect();
        if found.is_empty() {
            bail!("I don't have a memory numbered #{id}");
        }
        return Ok(found);
    }
    let words = terms(query);
    if words.is_empty() {
        bail!("forget what?");
    }
    let all_words: Vec<Memory> = all.iter().filter(|m| { let t = terms(&m.text); words.iter().all(|w| t.contains(w)) }).cloned().collect();
    if !all_words.is_empty() {
        return Ok(all_words);
    }
    match rank(&all, query).into_iter().next() {
        Some((_, m)) => Ok(vec![m]),
        None => bail!("I don't remember anything like that"),
    }
}

/// Remove memories (at most five at a time, so one careless word can't wipe the lot).
pub fn forget(query: &str) -> Result<Vec<Memory>> {
    let matches = forget_matches(query)?;
    if matches.len() > 5 {
        bail!("that matches {} memories; be more specific or use /forget #number", matches.len());
    }
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ids: HashSet<u64> = matches.iter().map(|m| m.id).collect();
    let kept: Vec<Memory> = read().into_iter().filter(|m| !ids.contains(&m.id)).collect();
    write(&kept)?;
    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    static ENV: Mutex<()> = Mutex::new(());

    fn fresh() -> (std::sync::MutexGuard<'static, ()>, PathBuf) {
        let guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("ojp-mem-{}-{}", std::process::id(), Instant::now().elapsed().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        unsafe { std::env::set_var("OJP_MEMORY_FILE", dir.join("memory.jsonl")) };
        (guard, dir)
    }

    #[test]
    fn saves_lists_and_does_not_duplicate() {
        let (_g, dir) = fresh();
        let a = add("My sister is called Mia.").unwrap();
        assert_eq!(a.id, 1);
        assert_eq!(a.text, "My sister is called Mia");
        let again = add("my sister is called MIA").unwrap();
        assert_eq!(again.id, 1, "same fact twice is one memory");
        let b = add("I work at Initech on Tuesdays").unwrap();
        assert_eq!(b.id, 2);
        assert_eq!(list().len(), 2);
        assert!(add("hi").is_err() && add(&"x ".repeat(200)).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.join("memory.jsonl")).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "private to the user");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn search_ranks_by_relevance() {
        let (_g, dir) = fresh();
        for t in ["My sister is called Mia", "I work at Initech on Tuesdays", "I am allergic to peanuts", "My dentist appointment is on Friday"] {
            add(t).unwrap();
        }
        assert_eq!(search("What's my sister's name?", 3)[0].1.text, "My sister is called Mia");
        assert_eq!(search("where do I work", 3)[0].1.text, "I work at Initech on Tuesdays");
        assert_eq!(search("peanut allergy?", 3)[0].1.text, "I am allergic to peanuts");
        assert!(search("quantum chromodynamics", 3).is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn small_memories_are_all_given_to_the_model() {
        let (_g, dir) = fresh();
        assert!(context_for("anything").is_none());
        add("I love green tea").unwrap();
        add("My sister is called Mia").unwrap();
        let c = context_for("What do I like to drink?").unwrap();
        assert!(c.contains("green tea") && c.contains("Mia"), "{c}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn large_memories_give_the_relevant_ones_and_the_newest() {
        let (_g, dir) = fresh();
        for i in 0..30 {
            add(&format!("fact number {i} about topic{i}")).unwrap();
        }
        add("My sister is called Mia").unwrap();
        for i in 30..40 {
            add(&format!("fact number {i} about topic{i}")).unwrap();
        }
        let c = context_for("what is my sister called").unwrap();
        assert!(c.contains("Mia"), "relevant one is included: {c}");
        assert!(c.contains("topic39"), "newest is included: {c}");
        assert!(c.lines().count() < 14, "stays short: {}", c.lines().count());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn forgetting_is_exact_and_limited() {
        let (_g, dir) = fresh();
        add("My sister is called Mia").unwrap();
        add("I work at Initech").unwrap();
        add("My brother is called Leo").unwrap();
        assert_eq!(forget_matches("#2").unwrap()[0].text, "I work at Initech");
        assert!(forget_matches("#9").is_err());
        let gone = forget("sister Mia").unwrap();
        assert_eq!(gone.len(), 1);
        assert_eq!(list().len(), 2);
        assert!(forget("zebra").is_err());
        // an over-broad word cannot wipe everything
        for i in 0..7 {
            add(&format!("note {i} called thing{i}")).unwrap();
        }
        assert!(forget("called").unwrap_err().to_string().contains("matches"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn similar_words_share_one_tag() {
        let (_g, dir) = fresh();
        assert_eq!(tag_for_word("baking").as_deref(), Some("cooking"));
        assert_eq!(tag_for_word("Recipes").as_deref(), Some("cooking"));
        assert_eq!(tag_for_word("dinner").as_deref(), Some("cooking"));
        assert_eq!(tag_for_word("appointments").as_deref(), Some("schedule"));
        assert_eq!(tag_for_word("zebra"), None);
        assert!(auto_tags("I love baking sourdough bread").contains(&"cooking".to_string()));
        assert_eq!(auto_tags("My sister is called Mia"), vec!["people"]);
        assert_eq!(auto_tags("The sky is blue"), vec![OTHER]);
        assert!(auto_tags("My dentist appointment is on Friday").contains(&"schedule".to_string()));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn memories_are_filed_and_found_by_tag() {
        let (_g, dir) = fresh();
        let m = add("I make pasta for dinner on Sundays").unwrap();
        assert!(m.tags.contains(&"cooking".to_string()), "{:?}", m.tags);
        add("My sister is called Mia").unwrap();
        // The question shares no word with the memory, only the tag.
        let found = search("any baking tips for me?", 3);
        assert_eq!(found[0].1.text, "I make pasta for dinner on Sundays");
        assert_eq!(with_tag("cooking").len(), 1);
        assert_eq!(with_tag("people").len(), 1);
        let counts = tag_counts();
        assert_eq!(counts.iter().find(|(t, _)| t == "cooking").unwrap().1, 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn long_memories_get_a_short_gist() {
        let (_g, dir) = fresh();
        let long = "My daughter has a recital at the community hall on the fifteenth of March, and I promised to bring her cello and a thermos of tea before it starts.";
        let m = add(long).unwrap();
        assert!(m.summary.chars().count() <= 91 && m.summary.starts_with("My daughter has a recital"), "{}", m.summary);
        assert_eq!(m.gist(), m.summary);
        let short = add("I love green tea").unwrap();
        assert_eq!(short.gist(), "I love green tea");
        // The context for the model carries the gist, not the long text.
        let c = context_for("recital").unwrap();
        assert!(c.contains("My daughter has a recital") && !c.contains("thermos"), "{c}");
        // The model (or the user) can refine the filing afterwards.
        set_meta(m.id, vec!["people".into(), "schedule".into()], "Daughter's cello recital, 15 March".into()).unwrap();
        assert_eq!(list().iter().find(|x| x.id == m.id).unwrap().gist(), "Daughter's cello recital, 15 March");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn users_can_add_tags_but_not_silly_ones() {
        let (_g, dir) = fresh();
        add_vocabulary("gardening", &["plants".into(), "Flowers".into()]).unwrap();
        assert_eq!(tag_for_word("flower").as_deref(), Some("gardening"));
        assert!(auto_tags("I water my plants every morning").contains(&"gardening".to_string()));
        add_vocabulary("cooking", &["grilling".into()]).unwrap();
        assert_eq!(tag_for_word("grilling").as_deref(), Some("cooking"));
        assert!(add_vocabulary("two words", &[]).is_err() && add_vocabulary("", &[]).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
