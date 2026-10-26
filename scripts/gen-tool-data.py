#!/usr/bin/env python3
"""Writes ojp-core/data/tool-train.tsv: <tool or "multi"><TAB><request>, for fitting
the tool probe. Templated phrasing per tool with slot variety (politeness, casing,
fillers). The phrasings in `ojp-cli tool-eval` (TOOL_CASES, UNSEEN, MULTI) are never
copied here: they are the held-out test, and exact duplicates are dropped."""
import random, re
from pathlib import Path

random.seed(11)
root = Path(__file__).resolve().parent.parent
src = (root / "ojp-core/src/bin/ojp-cli.rs").read_text()
held = {m.lower() for m in re.findall(r'\("((?:[^"\\]|\\.)*)", "[a-z_]+", Check::', src)}
held |= {m.lower() for m in re.findall(r'\("((?:[^"\\]|\\.)*)", &\[', src)}

N = lambda: random.choice([2, 3, 5, 7, 8, 10, 12, 15, 20, 25, 30, 40, 45, 60, 90])
V = lambda: random.choice([5, 10, 15, 20, 25, 30, 35, 45, 50, 60, 65, 70, 75, 85, 90, 100])
apps = ["Firefox", "Chrome", "Spotify", "the terminal", "Slack", "Discord", "Steam", "VS Code", "the calculator", "the file manager", "Thunderbird", "Telegram", "Blender", "GIMP", "the settings", "LibreOffice", "OBS", "Kate", "Dolphin", "Signal"]
sites = ["github.com", "wikipedia.org", "news.ycombinator.com", "youtube.com", "reddit.com", "archlinux.org", "duckduckgo.com", "openstreetmap.org", "docs.rs", "gitlab.com"]
things = ["buy milk", "call the dentist", "pick up the parcel", "water the plants", "email Jane the report", "renew the passport", "book a table for Friday", "check the oven", "feed the cat", "pay the electricity bill", "stretch", "take the bins out", "submit the form", "back up the laptop"]
files = ["tax return", "resume", "budget", "passport scan", "holiday photos", "invoice", "thesis", "contract", "recipe", "receipt", "lecture notes", "wedding photos", "insurance policy", "cv", "boarding pass"]
words = ["wifi password", "passport number", "rent", "salary", "bank account", "meeting", "deadline", "phone number", "address", "flight", "doctor", "insurance", "budget", "password", "birthday"]
dirs = ["Documents", "Downloads", "Pictures", "Music", "Videos", "Desktop", "my home folder", "the projects folder"]
fnames = ["notes.txt", "todo.txt", "budget.csv", "readme.md", "draft.txt", "shopping.txt", "ideas.md", "log.txt", "report.txt", "plan.md"]
facts = ["my sister is called Mia", "I'm allergic to peanuts", "I work night shifts", "my favourite colour is green", "I live in Denver", "I'm learning the guitar", "my dog is called Max", "I don't eat pork", "my birthday is in June", "I drive a blue car", "my manager is Priya", "I prefer tea to coffee"]
topics = ["allergies", "my dog", "my job", "food", "where I live", "my family", "my hobbies", "drinks", "my car", "my birthday"]
cmds = ["ls -la", "df -h", "uptime", "whoami", "date", "free -m", "pwd", "git status", "ps aux", "uname -a", "ls ~/Downloads", "cat /etc/os-release"]
queries = ["rust tutorials", "the weather in Denver", "cheap flights to Rome", "best hiking boots", "how to bake bread", "python decorators explained", "latest linux kernel", "pizza near me", "train times to Boston", "movie showtimes tonight"]
exprs = [("17 times 23", ""), ("15% of 240", ""), ("the square root of 144", ""), ("1234 divided by 8", ""), ("2 to the power of 10", ""), ("45 plus 38", ""), ("999 minus 457", ""), ("12 squared", ""), ("7 times 8 plus 3", ""), ("3.5 times 4", ""), ("100 divided by 7", ""), ("18% tip on 64 dollars", ""), ("25 percent of 480", ""), ("sine of 30 degrees", ""), ("the cube of 9", "")]
tags = ["gardening", "travel", "fitness", "finance", "music", "work", "school", "family"]

T = {
"set_timer": ["set a timer for {n} minutes", "start a {n} minute timer", "{n} minute timer please", "timer for {n} minutes", "count down {n} minutes", "wake me in {n} minutes", "give me a {n} minute countdown", "ring after {n} minutes", "start the egg timer, {n} minutes", "can you time {n} minutes for me", "let me know when {n} minutes are up", "begin a countdown of {n} minutes"],
"remind": ["remind me to {t} in {n} minutes", "in {n} minutes remind me to {t}", "ping me in {n} minutes about {t}", "tell me to {t} after {n} minutes", "set a reminder to {t} in {n} minutes", "don't let me forget to {t}, {n} minutes from now", "reminder: {t}, {n} minutes", "nudge me in {n} minutes to {t}"],
"volume": ["set the volume to {v}", "volume {v}", "turn the volume up to {v}", "turn the volume down to {v} percent", "make it louder, {v}", "put the sound at {v}", "adjust the speaker volume to {v}", "volume to {v} percent please", "lower the volume to {v}", "raise the volume to {v}", "change the volume level to {v}", "can you make the volume {v}"],
"mute": ["mute the sound", "mute", "unmute the speakers", "unmute", "silence the speakers", "turn the sound off", "turn the sound back on", "shut the speakers up", "mute my computer", "bring the sound back", "kill the audio", "restore the sound", "mute it", "stop the noise, mute everything"],
"media": ["pause the music", "play the music", "skip this song", "next song", "next track please", "previous track", "go back to the last song", "resume the music", "pause the video", "stop the song for a second", "skip to the next one", "play the previous song again", "hit play", "hit pause", "play or pause"],
"open_app": ["open {a}", "launch {a}", "start {a}", "bring up {a}", "run {a}", "fire up {a}", "can you open {a}", "switch to {a}", "get {a} going", "open up {a} for me"],
"open_url": ["open {s} in the browser", "go to {s}", "take me to {s}", "show me {s}", "navigate to https://{s}", "open https://{s}", "pull up {s} in my browser", "visit {s}", "load the website {s}", "browse to {s}"],
"note_add": ["take a note: {t}", "note to self: {t}", "write down: {t}", "save a note that says {t}", "jot down {t}", "make a note: {t}", "add a note: {t}", "put this in my notes: {t}", "note that I need to {t}", "scribble down {t}"],
"calculate": ["what's {e}", "calculate {e}", "how much is {e}", "what is {e}", "work out {e}", "compute {e}", "quick maths: {e}", "can you tell me {e}", "{e}?", "figure out {e}"],
"get_time": ["what time is it", "what's the time", "tell me the time", "what's today's date", "what day is it", "what's the date", "current time please", "do you know what time it is", "what day of the week is it", "what's the date today", "time check", "what's today"],
"system_status": ["how much battery do I have", "how's my battery", "how much disk space is free", "how much free space do I have", "how much ram is free", "how much memory is used", "what's my cpu load", "how long has the computer been on", "is the laptop plugged in", "how's my computer doing", "check system status", "how full is my disk", "what's the uptime", "any low resources on this machine"],
"now_playing": ["what song is playing", "what's playing", "what am I listening to", "which track is this", "what's the current song", "what song is this", "who sings this", "name of the song playing right now", "what's on the speakers", "what video is playing"],
"clipboard": ["what's on my clipboard", "read my clipboard", "what did I copy", "show me what I copied", "paste the clipboard contents here", "what's in the clipboard", "tell me my clipboard text", "check the clipboard"],
"find_files": ["find my {f} files", "where is my {f}", "locate my {f}", "search for files named {f}", "find the file called {f}", "look for my {f}", "which file is my {f}", "find {f} on this computer", "where did I save the {f}", "help me find my {f}"],
"reminders_list": ["what reminders do I have", "show my reminders", "what timers are running", "list my pending reminders", "do I have any timers", "are there any reminders set", "what's pending on my timers", "show active timers and reminders", "any reminders coming up", "what reminders are pending"],
"list_dir": ["what files are in {d}", "list the files in {d}", "show me what's in {d}", "what's in my {d} folder", "ls {d}", "show the contents of {d}", "list {d}", "what do I have in {d}", "how many files are in {d}", "browse {d}"],
"search_text": ["search my files for {w}", "which of my files mentions {w}", "check my documents and tell me {w}", "look through my files for {w}", "find where I wrote down {w}", "what do my documents say about {w}", "grep my notes for {w}", "check my files: what is my {w}", "search inside my documents for {w}", "dig through my files for the {w}"],
"read_file": ["read {fn} in my Documents", "show me what's inside {fn}", "open and read ~/{fn}", "what does {fn} say", "cat {fn}", "read the file {fn}", "display the contents of {fn}", "print {fn}", "tell me what's written in {fn}", "read me {fn}"],
"write_file": ["save the text {t} into {fn} in my Documents", "create a file called {fn} containing {t}", "write {t} to {fn}", "make a new file {fn} with the text {t}", "put {t} into a file named {fn}", "create {fn} and write {t} in it", "store {t} in {fn}", "overwrite {fn} with {t}"],
"run_command": ["run {c} in the terminal", "run the command {c}", "execute {c}", "run {c}", "in the shell run {c}", "type {c} in the terminal and show me the output", "can you run {c}", "use the terminal to run {c}"],
"web_search": ["search the web for {q}", "google {q}", "look up {q} online", "search online for {q}", "find information about {q} on the internet", "web search: {q}", "look {q} up on the web", "can you search for {q}"],
"web_fetch": ["read the page at https://{s}", "fetch https://{s} and tell me what it says", "what does {s} say", "summarize the page https://{s}", "read the article at {s} to me", "get the text of {s}", "go to {s} and tell me what it says"],
"remember": ["remember that {x}", "please remember {x}", "keep in mind that {x}", "save this about me: {x}", "note for the future: {x}", "I want you to remember {x}", "store this fact: {x}", "don't forget that {x}"],
"recall": ["what did I tell you about {p}", "what do you know about {p}", "look up what I told you about {p}", "what do I like to drink", "what is my dog called", "do you remember {p}", "recall what I said about {p}", "what's my favourite colour", "where do I live", "who is my manager"],
"memories_list": ["what do you remember about me", "show everything you remember", "list my memories", "what have I told you so far", "what do you know about me", "show me all the memories", "tell me everything you remember about me"],
"tags_list": ["which tags do you use for my memories", "list the memory tags", "what tags exist", "show the memory categories", "what tags are there for memories"],
"tag_add": ["add a tag called {g} for {w}", "create a memory tag {g} with words {w}", "make a new tag named {g} for {w}", "add the words {w} to the {g} tag", "new memory tag {g}: {w}"],
"forget": ["forget that {x}", "forget what I told you about {p}", "remove {p} from your memory", "delete the memory about {p}", "please forget {x}", "erase what you know about {p}", "stop remembering {p}"],
"notes_list": ["what notes do I have", "show me what I wrote down", "read my notes back to me", "list my saved notes", "show my notes", "what's in my notes", "read me my notes", "what did I jot down"],
}
# Words that tag_add 'words' slot may take.
wordlists = ["plants, flowers", "running, gym, yoga", "flights, hotels", "salary, rent, bills", "guitar, piano", "meeting, boss, project", "exam, homework", "mum, dad, cousins"]

def fill(t):
    e, _ = random.choice(exprs)
    return t.format(n=N(), v=V(), t=random.choice(things), a=random.choice(apps), s=random.choice(sites), f=random.choice(files), w=random.choice(wordlists) if "{g}" in t else random.choice(words), d=random.choice(dirs), fn=random.choice(fnames), c=random.choice(cmds), q=random.choice(queries), x=random.choice(facts), p=random.choice(topics), e=e, g=random.choice(tags))

pre = ["", "", "", "", "please ", "can you ", "could you ", "hey, ", "hey jarvis, ", "ok ", "i want you to "]
post = ["", "", "", ".", "?", " please", "!", " for me", " now"]
def dress(s):
    s = random.choice(pre) + s + random.choice(post)
    if random.random() < 0.1: s = s.lower()
    return s[0].upper() + s[1:] if random.random() < 0.7 else s

rows = set()
per_tool = 42
for tool, tpls in T.items():
    n = 0
    tries = 0
    while n < per_tool and tries < 2000:
        tries += 1
        r = dress(fill(random.choice(tpls))).strip()
        if r.lower() in held or (tool, r) in rows: continue
        rows.add((tool, r)); n += 1
# Two-step requests: the first words are one tool, the rest another.
singles = [t for t in T if t not in ("write_file", "tag_add", "forget", "web_fetch", "memories_list", "tags_list")]
joiners = [" and ", ", then ", " and then ", " and also ", ", and "]
for _ in range(220):
    a, b = random.sample(singles, 2)
    ra = fill(random.choice(T[a])).rstrip(".?!"); rb = fill(random.choice(T[b])).rstrip(".?!")
    r = dress(ra + random.choice(joiners) + rb[0].lower() + rb[1:])
    if r.lower() in held: continue
    rows.add(("multi", r))
out = sorted(rows)
random.shuffle(out)
(root / "ojp-core/data/tool-train.tsv").write_text("".join(f"{t}\t{r}\n" for t, r in out))
from collections import Counter
print(len(out), "rows;", len(Counter(t for t, _ in out)), "classes ->", "ojp-core/data/tool-train.tsv")
# Held-out file: single-tool cases from tool-eval (name = expected tool) plus the multi-step cases (label multi when 2+ tools).
test = []
for m in re.finditer(r'\("((?:[^"\\]|\\.)*)", "([a-z_]+)", Check::', src):
    test.append((m.group(2), m.group(1)))
for m in re.finditer(r'\("((?:[^"\\]|\\.)*)", &\[([^\]]*)\]\)', src):
    tools = re.findall(r'"([a-z_]+)"', m.group(2))
    test.append((tools[0] if len(tools) == 1 else "multi", m.group(1)))
(root / "ojp-core/data/tool-test.tsv").write_text("".join(f"{t}\t{r}\n" for t, r in test))
print(len(test), "held-out rows -> ojp-core/data/tool-test.tsv")
