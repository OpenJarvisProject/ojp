#!/usr/bin/env python3
"""Soak test for a model bundle through the real engine (ojp-cli chat): start-up time (cold and with the saved
cache), latency per kind of request, RSS growth over a long mixed conversation, odd input (emoji, other
languages, very long lines), and a scan of the replies for think-tag leaks or empty answers.

  final-checks.py <config.toml> [messages]   (run from the ojp repo; needs target/release/ojp-cli)"""
import os, subprocess, sys, tempfile, time, re, json, shutil
from pathlib import Path

cfg = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
tmp = Path(tempfile.mkdtemp(prefix="ojp-final-"))
env = {**os.environ, "HOME": str(tmp), "XDG_DATA_HOME": str(tmp / "data"), "XDG_CONFIG_HOME": str(tmp / "cfg"),
       "OJP_NOTES_FILE": str(tmp / "notes.md"), "OJP_REMINDERS_FILE": str(tmp / "reminders.json"),
       "OJP_MEMORY_FILE": str(tmp / "memory.jsonl"), "OJP_TOOLS_DRY_RUN": "1", "OJP_N_CTX": "4096", "OJP_THREADS": "6"}
(tmp / "Documents").mkdir()
(tmp / "Documents/budget-2026.txt").write_text("Rent 1200, Food 400, Fun 150")
CLI = os.environ.get("OJP_CLI", "target/release/ojp-cli")

def rss(pid):
    try:
        return int(re.search(r"VmRSS:\s+(\d+)", open(f"/proc/{pid}/status").read()).group(1)) // 1024
    except Exception:
        return 0

class Chat:
    def __init__(self):
        t = time.time()
        self.p = subprocess.Popen([CLI, "chat", cfg], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(tmp / 'chat-stderr.log', 'ab'), env=env, bufsize=0)
        self.err = b""
        self.read_prompt()
        self.startup = time.time() - t
    def read_prompt(self):
        buf = b""
        while not buf.endswith(b"> "):
            c = self.p.stdout.read(1)
            if not c: break
            buf += c
        return buf[:-2].decode("utf-8", "replace")
    def ask(self, msg):
        t = time.time(); self.p.stdin.write((msg + "\n").encode()); self.p.stdin.flush()
        out = self.read_prompt()
        return out.strip(), time.time() - t, rss(self.p.pid)
    def close(self):
        try: self.p.stdin.write(b"\n"); self.p.stdin.flush(); self.p.wait(timeout=20)
        except Exception: self.p.kill()

report = {}
for label in ("cold start", "start with saved cache"):
    c = Chat(); report[label] = {"seconds": round(c.startup, 1), "rss_mb": rss(c.p.pid)}; c.close()

kinds = {
 "greeting": ["Hi!", "Good evening, how are you?", "thanks, that's all"],
 "fact": ["What's the capital of Canada?", "Why is the sky blue?", "Who painted the Mona Lisa?"],
 "roleplay": ["You are Luna, my warm, playful companion. I just got home. Greet me and ask about my day.", "*sits down next to you* I had a rough day at work.", "Tell me a bedtime story about a dragon, three sentences.", "Stay in character and tease me gently about being late."],
 "coding": ["Write a Python function that reverses a string.", "Why does my Rust code say 'borrowed value does not live long enough'?"],
 "reasoning": ["If a bat and ball cost $1.10 and the bat costs $1 more than the ball, how much is the ball?", "Help me plan a three-day trip to Rome on a small budget."],
 "tools": ["What time is it?", "Set a timer for 5 minutes.", "What's 17 times 23?", "Remember that my sister is called Mia.", "What is my sister called?", "How much is my rent? Look in my budget file.", "What files are in my Documents folder?"],
 "odd input": ["こんにちは、元気ですか？", "Привет! Расскажи короткую шутку.", "😀🔥 what does this emoji mean: 🫠", "¿Cuál es la capital de España?", "a" * 1500, "Résumé naïve café — dîner à l'œuf"],
}
detail = []
c = Chat(); rows = []; leaks = 0; empties = 0; sizes = []
msgs = [(k, m) for k, ms in kinds.items() for m in ms]
while len(msgs) < N: msgs += msgs[: N - len(msgs)]
for kind, m in msgs[:N]:
    out, dt, mem = c.ask(m)
    rows.append((kind, dt, mem, len(out))); detail.append((kind, m[:36], round(dt, 1), len(out)))
    leaks += bool(re.search(r"</?think>", out)); empties += (len(out) == 0)
    sizes.append(mem)
c.close()
by = {}
for k, dt, mem, n in rows: by.setdefault(k, []).append(dt)
report["latency seconds per request (mean / max)"] = {k: f"{sum(v)/len(v):.1f} / {max(v):.1f}" for k, v in by.items()}
report["rss mb: after 5th, middle, last request"] = [sizes[4], sizes[len(sizes)//2], sizes[-1]]
report["rss growth over the run (mb)"] = sizes[-1] - sizes[4]
report["slowest requests"] = sorted(detail, key=lambda d: -d[2])[:10]
report["think-tag leaks"] = leaks; report["empty replies"] = empties; report["requests"] = len(rows)
print(json.dumps(report, indent=1))
shutil.rmtree(tmp, ignore_errors=True)
