#!/usr/bin/env bash
# Format, lint, and test the whole workspace (mirrors pi's ./test.sh + npm run check).
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
