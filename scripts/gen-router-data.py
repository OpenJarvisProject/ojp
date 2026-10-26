#!/usr/bin/env python3
"""Writes ojp-core/data/router-train.tsv: label<TAB>request, for fitting the
router probe (scripts/fit-router-probe.py). Hand-written seeds are extracted
from ojp-cli.rs (TRAIN/DEV/ROUTER sets; XL_CASES is never included, it is the
held-out test), plus templated variety: politeness, casing, typos, topics."""
import random, re, sys
from pathlib import Path

random.seed(7)
root = Path(__file__).resolve().parent.parent
src = (root / "ojp-core/src/bin/ojp-cli.rs").read_text()
LABELS = {"AnswerDirectly": "answer_directly", "Coding": "coding", "Reasoning": "reasoning", "DeviceControl": "device_control"}

def block(name):
    m = re.search(r"const %s: \[\(&str, Route\); \d+\] = \[(.*?)\n\];" % name, src, re.S)
    return [(LABELS[l], q.encode().decode("unicode_escape")) for q, l in re.findall(r'\("((?:[^"\\]|\\.)*)", Route::(\w+)\)', m.group(1))]

rows = block("TRAIN_CASES") + block("DEV_CASES") + block("ROUTER_CASES")
rt = (root / "ojp-core/src/router.rs").read_text()
for q, l in re.findall(r'\("((?:[^"\\]|\\.)*)", Route::(\w+)\)', rt):
    rows.append((LABELS[l], q))

pre = ["", "", "", "please ", "can you ", "could you ", "hey, ", "hey jarvis, ", "ok ", "i want you to ", "would you "]
post = ["", "", ".", "?", " please", " for me", "!", " now"]
def dress(t, p=0.6):
    t = random.choice(pre) + t + random.choice(post) if random.random() < p else t
    if random.random() < 0.12: t = t.lower()
    if random.random() < 0.05: t = t.rstrip(".?!")
    return t[0].upper() + t[1:] if random.random() < 0.7 else t

topics = ["the Eiffel Tower", "black holes", "the French Revolution", "honey bees", "volcanoes", "the Pacific Ocean", "Mount Everest", "jazz", "the printing press", "dolphins", "the Amazon river", "Marie Curie", "chess", "the Great Wall of China", "rainbows", "the human heart", "Shakespeare", "the Sahara", "electricity", "Leonardo da Vinci", "tea", "the Olympics", "glaciers", "tigers", "the Moon landing", "Mars", "bread", "the Nile", "castles", "coffee"]
ans = ["what is {t}", "tell me about {t}", "who knows about {t}", "give me a quick fact about {t}", "why is {t} famous", "what's interesting about {t}", "explain {t} in one sentence", "how old is {t}", "where is {t}"]
chat = ["how are you today", "what's up", "you're awesome", "thanks a lot", "nice to meet you", "i'm bored", "tell me a joke", "what's your name", "good afternoon", "see you later", "that was helpful", "do you ever get tired", "i'm feeling a bit sad", "i had a great day", "how's the weather treating you"]
langs = ["Python", "JavaScript", "Rust", "Go", "Java", "C++", "TypeScript", "C#", "Ruby", "PHP", "Kotlin", "Swift", "Bash", "SQL", "Haskell"]
codeq = ["write a {l} function that {x}", "how do I {x} in {l}", "why does my {l} code {y}", "fix this {l} snippet that {y}", "show an example of {z} in {l}", "what is the idiomatic way to {x} in {l}", "explain {z} in {l}", "convert this loop to a one-liner in {l}", "write a unit test in {l} for a function that {x}", "debug: my {l} program {y}"]
xs = ["reverses a string", "sorts a list of numbers", "reads a CSV file", "checks if a number is prime", "removes duplicates from an array", "counts words in a file", "parses a date", "merges two dictionaries", "finds the max value", "validates an email address", "downloads a web page", "computes fibonacci numbers", "connects to a database", "flattens nested lists", "reads environment variables"]
ys = ["crashes with a null pointer", "never terminates", "prints the wrong output", "throws a type error", "leaks memory", "runs out of memory", "fails to compile", "returns undefined", "hangs on large input", "gives an off-by-one result"]
zs = ["recursion", "generics", "closures", "async await", "pattern matching", "iterators", "decorators", "inheritance", "lambdas", "error handling", "pointers", "threads", "regular expressions", "list comprehensions", "interfaces"]
reas = ["help me plan {a}", "what are the pros and cons of {b}", "should I {c} or {d}? think it through", "break down how to {e}", "work out how much {f}", "walk me through deciding whether to {c}", "compare {b} with {g}", "estimate {h}", "make a schedule for {a}", "figure out the best way to {e}"]
A = ["a weekend trip to the coast", "a wedding on a small budget", "my study time for finals", "a vegetarian meal plan for a week", "moving to a new apartment", "a home renovation", "a charity fundraiser", "a month of workouts", "a road trip across the country", "a garden for beginners"]
B = ["solar panels", "working from home", "adopting a dog", "a gap year", "buying a used car", "learning two languages at once", "remote learning", "renting vs owning", "freelancing", "public transport vs driving"]
C = ["take the new job", "move abroad", "go back to school", "buy a house", "start a business", "switch careers", "lease a car", "sell my old car", "get a roommate", "adopt a cat"]
E_ = ["save for retirement", "pay off student loans", "learn to cook", "get fit in three months", "start a podcast", "write a book in a year", "learn to code", "organize a big move", "prepare for a marathon", "build an emergency fund"]
F = ["I need to save each month to afford a $24000 car in two years", "a $1200 bill costs split five ways with 20% tip", "paint I need for a room that is 4 by 5 meters", "interest I earn on $5000 at 3% over 4 years", "time a 450 mile trip takes at 55 mph with two stops", "gas a 300 km drive uses at 7 liters per 100 km"]
G = ["renting", "buying", "walking", "the train", "a laptop", "a tablet", "cash", "credit", "a big city", "a small town"]
H = ["how many tennis balls fit in a bathtub", "how many words are in a typical novel", "how many cups of coffee a city drinks per day", "how long it takes to count to a million", "how many people are in the air right now"]
RIDDLE = ["If a clock takes 6 seconds to strike 6, how long does it take to strike 12? Explain.", "I have two kids and one is a boy; what's the chance the other is a boy? Reason it out.", "A snail climbs 3 feet by day and slips 2 at night in a 10 foot well; when does it get out?", "Which weighs more, a pound of feathers or a pound of bricks? Explain why people get this wrong.", "If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?"]
dev_t = ["{n} minutes", "{n} seconds", "an hour", "half an hour"]
dev = ["set a timer for {tm}", "start a countdown of {tm}", "remind me to {r} in {tm}", "wake me up at {h}", "set an alarm for {h}", "remind me to {r} at {h}", "turn the volume {vd}", "set the volume to {v} percent", "mute the sound", "unmute", "make the screen {bd}", "set brightness to {v}", "open {app}", "launch {app}", "close {app}", "switch to {app}", "turn {onoff} {sw}", "toggle {sw}", "{mus}", "what's the time", "what time is it in {city}", "how's my battery", "how much storage is left", "what's on my clipboard", "note that {r}", "save a note: {r}", "read my notes", "list my notes", "show the files in {fold}", "find the file {fn}", "open {fn}", "read {fn} to me", "create a file named {fn} containing {r}", "delete {fn}", "search the web for {q}", "look up {q} online", "google {q}", "what's {a} {op} {b}", "calculate {a} {op} {b}", "run the command {cmd}", "remember that {fact}", "forget that {fact}", "what do you remember about me", "what are my reminders", "cancel my timer", "what's playing"]
rr = ["call mom", "water the plants", "take out the trash", "submit the report", "buy milk", "email the landlord", "stretch", "pay rent", "feed the cat", "check the oven"]
hh = ["6 am", "7:30", "noon", "9 pm", "8:15 tomorrow", "5:45 am", "midnight"]
apps = ["Firefox", "Spotify", "the terminal", "Slack", "the file manager", "Discord", "Steam", "VS Code", "the calculator", "the settings app", "Telegram", "Thunderbird"]
sws = ["Wi-Fi", "Bluetooth", "dark mode", "airplane mode", "night light", "do not disturb", "the flashlight", "location services"]
mus = ["play some jazz", "pause the music", "skip this song", "next track", "previous song", "play my liked songs", "stop the music", "resume playback", "shuffle my playlist", "what song is this"]
cities = ["Tokyo", "London", "New York", "Sydney", "Paris", "Berlin", "Toronto", "Dubai"]
folds = ["my Downloads folder", "my Documents folder", "the Pictures folder", "my home directory", "the Music folder"]
fns = ["notes.txt", "todo.txt", "budget.csv", "resume.pdf", "shopping.txt", "ideas.md", "taxes.xlsx", "report.docx", "draft.txt", "contacts.txt"]
qs = ["the weather in Chicago", "best hiking trails nearby", "how to bake sourdough", "cheap flights to Rome", "movie showtimes tonight", "latest tech news", "reviews of the new phone", "train times to Boston", "pizza places near me", "today's stock market"]
cmds = ["ls -la", "df -h", "uptime", "whoami", "date", "free -m", "pwd", "git status", "ps aux", "uname -a"]
facts = ["I'm allergic to peanuts", "my wife's name is Sarah", "I work night shifts", "my favorite color is green", "I live in Denver", "I'm learning guitar", "my dog is called Max", "I don't eat pork", "my birthday is in June", "I drive a blue car"]

def pick(seq): return random.choice(seq)
gen = []
for _ in range(110):
    gen.append(("answer_directly", dress(pick(ans).format(t=pick(topics)), 0.4)))
for c in chat:
    gen.append(("answer_directly", dress(c, 0.3)))
for _ in range(110):
    gen.append(("coding", dress(pick(codeq).format(l=pick(langs), x=pick(xs), y=pick(ys), z=pick(zs)), 0.5)))
for _ in range(100):
    gen.append(("reasoning", dress(pick(reas).format(a=pick(A), b=pick(B), c=pick(C), d=pick(C), e=pick(E_), f=pick(F), g=pick(G), h=pick(H)), 0.5)))
for r in RIDDLE:
    gen.append(("reasoning", r))
for _ in range(160):
    t = pick(dev).format(tm=pick(dev_t).format(n=random.choice([2, 5, 10, 15, 20, 30, 45, 90])), r=pick(rr), h=pick(hh), vd=pick(["up", "down", "up a bit", "down a lot"]), v=random.choice([10, 20, 35, 50, 75, 90]), bd=pick(["brighter", "dimmer"]), app=pick(apps), onoff=pick(["on", "off"]), sw=pick(sws), mus=pick(mus), city=pick(cities), fold=pick(folds), fn=pick(fns), q=pick(qs), a=random.randint(2, 999), b=random.randint(2, 99), op=pick(["plus", "minus", "times", "divided by"]), cmd=pick(cmds), fact=pick(facts))
    gen.append(("device_control", dress(t, 0.5)))
webq = ["rust tutorials", "python decorators", "react hooks guide", "docker compose examples", "git rebase tutorial", "css grid cheat sheet", "linux commands list", "java streams examples", "kubernetes basics", "sql join types", "typescript generics", "bash scripting tips"]
for _ in range(40):
    q = pick(webq)
    gen.append(("device_control", dress(pick(["search the web for {q}", "google {q}", "look up {q} online", "find {q} on the internet", "web search: {q}", "search online for {q}"]).format(q=q), 0.4)))
filesx = ["resume", "budget", "recipes", "tax return", "lease", "insurance policy", "cover letter", "notes file", "meeting notes", "contract", "todo list", "grocery list", "passport info"]
for _ in range(60):
    f = pick(filesx)
    gen.append(("device_control", dress(pick(["what does my {f} say", "what's in my {f}", "summarize my {f}", "check my {f} for me", "open my {f} and tell me what it says", "what is written in my {f}", "look in my {f} and tell me the details", "read me my {f}"]).format(f=f), 0.4)))
likes = ["drink", "eat for breakfast", "watch on weekends", "listen to", "do on holidays", "cook"]
things = ["dog", "cat", "sister", "boss", "best friend", "favourite colour", "car"]
for _ in range(45):
    gen.append(("device_control", dress(pick(["what do I like to {l}", "what is my {t} called", "what's my {t}'s name", "where do I live", "what did I tell you about my {t}", "do you remember my {t}", "who is my {t}", "what do you know about my {t}"]).format(l=pick(likes), t=pick(things)), 0.4)))
spend = ["food", "rent", "fun", "groceries", "transport", "subscriptions"]
for _ in range(45):
    gen.append(("device_control", dress(pick(["how much do I spend on {x} each month", "what's my monthly {x} budget", "how much is my {x} per month", "what is my rent", "what's my salary", "what is my wifi password", "when is my dentist appointment", "when is my flight"]).format(x=pick(spend)), 0.4)))
seen = set(); out = []
for l, q in rows + gen:
    k = q.strip().lower()
    if k in seen: continue
    seen.add(k); out.append((l, q.strip()))
random.shuffle(out)
path = root / "ojp-core/data/router-train.tsv"
path.write_text("".join(f"{l}\t{q}\n" for l, q in out))
from collections import Counter
print(len(out), Counter(l for l, _ in out), "->", path)
xl = block("XL_CASES")
(root / "ojp-core/data/router-xl.tsv").write_text("".join(f"{l}\t{q}\n" for l, q in xl))
print(len(xl), "held-out cases -> ojp-core/data/router-xl.tsv")
