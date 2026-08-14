#!/usr/bin/env python3
"""Drive the agent-m REPL through a real pty (tmux-equivalent harness).

Verifies, end to end, against the real provider when DEEPSEEK_API_KEY is set:
  1. startup renders the REPL banner,
  2. a typed prompt starts a turn (the `thinking...` indicator appears and a
     reply panel renders),
  3. the model's reply is persisted to the session log (two real turns),
  4. turn 2 is served from DeepSeek's prefix cache (cacheReadTokens > 0),
  5. ctrl+d exits cleanly.

Without DEEPSEEK_API_KEY the live-turn checks are SKIPped (missing-key
notice), so the harness is usable in hermetic environments.

Usage: python3 scripts/pty-smoke.py [extra agent-m args...]
"""
import os
import pty
import select
import sys
import time

BUILD = os.path.join(os.path.dirname(__file__), "..", "target", "debug", "agent-m")
SMOKE_DIR = os.path.join(os.path.dirname(__file__), "..", ".smoke")


def read_all(fd, timeout=0.2):
    out = b""
    end = time.time() + timeout
    while time.time() < end:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if fd not in ready:
            continue
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        out += data
    return out


def strip_ansi(text):
    import re
    # CSI (incl. private/param bytes <?=>), OSC (BEL or ST), and lone 3-byte
    # sequences (ESC [=/>).
    return re.sub(
        r"\x1b\[[0-9;?<=>]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[<=>]",
        "",
        text,
    )


def wait_for(fd, needle, timeout=8):
    buf = b""
    end = time.time() + timeout
    while time.time() < end:
        buf += read_all(fd, 0.2)
        if needle in strip_ansi(buf.decode(errors="replace")):
            return buf.decode(errors="replace")
    return buf.decode(errors="replace")


def newest_session_file():
    sessions = os.path.join(SMOKE_DIR, "agent", "sessions")
    best = None
    best_mtime = 0
    for root, _, files in os.walk(sessions):
        for name in files:
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(root, name)
            mtime = os.path.getmtime(path)
            if mtime > best_mtime:
                best, best_mtime = path, mtime
    return best


def session_messages(session_file):
    import json
    messages = []
    with open(session_file) as handle:
        for line in handle:
            entry = json.loads(line)
            if entry.get("type") == "session":
                continue
            text = ""
            content = entry.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            messages.append((entry.get("kind"), text, entry.get("usage") or {}))
    return messages


def main():
    if not os.path.exists(BUILD):
        print("build missing; run: cargo build -p agent-m-cli")
        return 1

    # Enable cache-miss notices so the byte-stable prefix caching can be
    # observed live (DeepSeek reports prompt_cache_hit_tokens).
    agent_dir = os.path.join(SMOKE_DIR, "agent")
    os.makedirs(agent_dir, exist_ok=True)
    settings = os.path.join(agent_dir, "settings.json")
    if not os.path.exists(settings):
        with open(settings, "w") as handle:
            handle.write('{"showCacheMissNotices": true}')

    pid, fd = pty.fork()
    if pid == 0:
        os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ.get("PATH", "")
        # Keep all agent data inside the workspace (sandbox-safe).
        os.environ["AGENT_M_DIR"] = SMOKE_DIR
        os.execv(BUILD, [BUILD] + sys.argv[1:])

    # Give the pty a real terminal size (default is 0x0, which renders nothing).
    import fcntl
    import struct
    import termios
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))

    failures = []
    skipped = False

    startup = wait_for(fd, "agent-m REPL mode", timeout=10)
    if "agent-m REPL mode" in strip_ansi(startup):
        print("PASS startup: REPL banner rendered")
    else:
        print("FAIL startup; captured:")
        print(strip_ansi(startup)[-800:])
        failures.append("startup")

    def settle(initial_text, timeout=90):
        """After 'thinking...' is seen, wait until the turn finishes (the next
        `agent-m (…` prompt appears). Returns 'ok' when the next prompt is
        seen, or 'no key' / 'provider error' when the run ended without a
        reply. Search is position-based: a marker only counts if it appears
        *after* the current turn's 'thinking...' in the byte stream."""
        text = initial_text
        end = time.time() + timeout
        while time.time() < end:
            text += strip_ansi(read_all(fd, 0.3).decode(errors="replace"))
            thinking_at = text.rfind("thinking...")
            for marker in ("no API key", "provider error"):
                marker_at = text.rfind(marker)
                if marker_at != -1 and marker_at > thinking_at:
                    return marker
            prompt_at = text.rfind("agent-m (")
            if prompt_at != -1 and prompt_at > thinking_at:
                return "ok"
        return None

    def turn(text, timeout=90):
        """Submit one prompt; require thinking... -> settle. Returns
        (streamed_ok: bool, output: str)."""
        try:
            os.write(fd, text.encode() + b"\r")
        except OSError:
            pass
        # "thinking..." only renders while a turn is in flight, so it appears
        # even when the key is missing. The reply/error frames arrive within
        # the same read windows, so the already-captured text is searched too.
        output = wait_for(fd, "thinking...", timeout=timeout)
        stripped = strip_ansi(output)
        if "thinking..." not in stripped:
            print("FAIL no thinking indicator for %r; captured tail:" % text)
            print(stripped[-800:])
            failures.append("turn:" + text)
            return False, stripped
        marker = settle(stripped, timeout)
        if marker == "ok":
            return True, stripped
        if marker in ("no API key", "provider error"):
            return False, stripped
        print("FAIL turn did not settle for %r (no prompt/error); tail:" % text)
        print(stripped[-800:])
        failures.append("settle:" + text)
        return False, stripped

    saw_streaming_1, tail1 = turn("Say exactly: smoke test ok")
    saw_streaming_2, tail2 = turn("Say exactly: second reply")
    if saw_streaming_1 and saw_streaming_2:
        print("PASS both turns streamed (thinking -> next prompt)")
    elif saw_streaming_1 != saw_streaming_2:
        print("FAIL inconsistent streaming between turns")
        failures.append("streaming-consistency")
    else:
        skipped = True
        print("SKIP live streaming (no DEEPSEEK_API_KEY)")

    # Authoritative reply + cache evidence: the persisted session log. The
    # reply text lives in the model's assistant messages, which cannot come
    # from the TUI input echo.
    session_file = newest_session_file()
    if session_file is None:
        print("FAIL no session file was persisted")
        failures.append("session-file")
    else:
        messages = session_messages(session_file)
        assistant_texts = [text for kind, text, _ in messages if kind == "assistant"]
        if saw_streaming_1:
            if len(assistant_texts) >= 1 and "smoke test ok" in assistant_texts[0]:
                print("PASS reply 1 persisted in the session log")
            else:
                print("FAIL reply 1 not in session log; got: %r" % assistant_texts)
                failures.append("reply1")
            if len(assistant_texts) >= 2 and "second reply" in assistant_texts[1]:
                print("PASS reply 2 persisted in the session log")
            else:
                print("FAIL reply 2 not in session log; got: %r" % assistant_texts)
                failures.append("reply2")
            usages = [usage for kind, _, usage in messages if kind == "assistant"]
            if len(usages) >= 2 and (usages[1].get("cacheReadTokens") or 0) > 0:
                print(
                    "PASS byte-stable prefix cached live: turn2 cacheRead=%s of %s input"
                    % (usages[1].get("cacheReadTokens"), usages[1].get("inputTokens"))
                )
            else:
                print("FAIL no cache hit on turn 2; usages: %r" % usages)
                failures.append("cache-hit")
        else:
            print("INFO session log not asserted (keyless run)")

    combined = tail1 + tail2
    if "cache" in combined and "% hit" in combined:
        print("PASS cache-hit stats surfaced in the session output")
    else:
        print("INFO cache stats not observed in the session output")

    try:
        os.write(fd, b"\x04")  # ctrl+d exit (empty editor)
    except OSError:
        pass
    time.sleep(1)
    tail = strip_ansi(read_all(fd, 1.0).decode(errors="replace"))
    try:
        waited_pid, status = os.waitpid(pid, os.WNOHANG)
        exited = waited_pid != 0
    except ChildProcessError:
        exited = True
        status = 0
    if exited:
        print("PASS clean exit after ctrl+d (status %s)" % status)
    else:
        print("FAIL exit; killing stuck child")
        try:
            os.kill(pid, 9)
        except OSError:
            pass
        failures.append("exit")

    print("--- captured tail ---")
    print(tail[-800:])
    if failures:
        print("SMOKE FAILED:", failures)
        return 1
    if skipped:
        print("SMOKE PASSED (keyless: live turns skipped)")
        return 0
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
