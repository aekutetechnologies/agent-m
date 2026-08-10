#!/bin/bash
# Structural validation for the Mintlify docs in docs/.
# Usage: scripts/docs-check.sh [--scaffold]  (run from the agent-m repo root)
# --scaffold skips the no-placeholders check (used before pages are filled).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FAIL=0
SCOPE="full"
if [ "${1:-}" = "--scaffold" ]; then
    SCOPE="scaffold"
fi

# 1) mint.json must be valid JSON.
if ! python3 -m json.tool docs/mint.json >/dev/null 2>&1; then
    echo "FAIL: docs/mint.json is not valid JSON"
    FAIL=1
else
    echo "ok: mint.json is valid JSON"
fi

# 2) every navigation path must resolve to a page file under docs/.
python3 - <<'PY'
import json, os, sys
nav = json.load(open("docs/mint.json"))["navigation"]
missing = []
def walk(items, group=""):
    for item in items:
        if isinstance(item, dict):
            walk(item.get("pages", []), item.get("group", group))
        elif isinstance(item, str):
            if not os.path.exists(f"docs/{item}.mdx"):
                missing.append(item)
for group in nav:
    walk([group])
sys.exit(1 if missing else 0)
PY
if [ $? -ne 0 ]; then
    echo "FAIL: navigation references pages that do not exist (see above)"
    FAIL=1
else
    echo "ok: every navigation path resolves to docs/<path>.mdx"
fi

# 3) internal markdown links (](/path) must resolve to docs/<path>.mdx.
python3 - <<'PY'
import os, re, sys
missing = []
for root, _dirs, files in os.walk("docs"):
    for name in files:
        if not name.endswith(".mdx"):
            continue
        path = os.path.join(root, name)
        text = open(path).read()
        for target in re.findall(r"\]\(/([a-z0-9\-/]+)\)", text):
            if not os.path.exists(f"docs/{target}.mdx"):
                missing.append(f"{path} -> /{target}")
sys.exit(1 if missing else 0)
PY
if [ $? -ne 0 ]; then
    echo "FAIL: some internal links do not resolve"
    FAIL=1
else
    echo "ok: all internal links resolve"
fi

# 4) code fences must be balanced (even count of ``` per file).
python3 - <<'PY'
import os, sys
bad = []
for root, _dirs, files in os.walk("docs"):
    for name in files:
        if not name.endswith(".mdx"):
            continue
        path = os.path.join(root, name)
        count = open(path).read().count("```")
        if count % 2 != 0:
            bad.append(path)
sys.exit(1 if bad else 0)
PY
if [ $? -ne 0 ]; then
    echo "FAIL: unbalanced code fences"
    FAIL=1
else
    echo "ok: code fences balanced"
fi

# 5) no file may be an unfilled placeholder (skipped in --scaffold scope).
if [ "$SCOPE" = "full" ] && grep -rq "# placeholder" docs --include="*.mdx"; then
    echo "FAIL: placeholder pages remain"
    FAIL=1
else
    echo "ok: no placeholder pages (scope=$SCOPE)"
fi

if [ "$FAIL" -ne 0 ]; then
    echo "docs-check FAILED"
    exit 1
fi
echo "docs-check PASSED"
