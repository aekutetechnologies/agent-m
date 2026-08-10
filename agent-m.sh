#!/usr/bin/env bash
# Run agent-m from source (mirrors pi's pi-test.sh).
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"

exec cargo run --quiet -p agent-m-cli -- "$@"
