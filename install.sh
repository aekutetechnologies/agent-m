#!/usr/bin/env sh
set -e

REPO="aekutetechnologies/agent-m"
BIN="agent-m"
INSTALL_DIR="${AGENT_M_INSTALL_DIR:-$HOME/.local/bin}"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)  PLATFORM="linux" ;;
  Darwin) PLATFORM="macos" ;;
  *)      echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

case "$ARCH" in
  x86_64)        TRIPLE="${PLATFORM}-x86_64" ;;
  arm64|aarch64) TRIPLE="${PLATFORM}-aarch64" ;;
  *)             echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

TAG="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
  | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')"

if [ -z "$TAG" ]; then
  echo "No release found. Install from source:" >&2
  echo "  cargo install --git https://github.com/${REPO} agent-m-cli" >&2
  exit 1
fi

URL="https://github.com/${REPO}/releases/download/${TAG}/agent-m-${TRIPLE}.tar.gz"

echo "Installing agent-m ${TAG} (${TRIPLE}) to ${INSTALL_DIR}..."
mkdir -p "$INSTALL_DIR"
curl -fsSL "$URL" | tar -xz -C "$INSTALL_DIR" "$BIN"
chmod +x "${INSTALL_DIR}/${BIN}"

echo ""
echo "agent-m ${TAG} installed to ${INSTALL_DIR}/${BIN}"
if ! echo "$PATH" | tr ':' '\n' | grep -qx "$INSTALL_DIR"; then
  echo "Add this to your shell profile:"
  echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
fi

if command -v cargo >/dev/null 2>&1; then
  echo "Installing Rust MCP servers (rust-mcp-filesystem, jira-mcp-rs, mcp-postgres)..."
  cargo install rust-mcp-filesystem jira-mcp-rs mcp-postgres 2>/dev/null || true
fi
