#!/usr/bin/env bash
# Sync .agents/ (source of truth) to .claude/ and .codex/
# Run from repo root: bash scripts/sync-ai-tools.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Syncing AI tools from .agents/ ..."

# Claude Code
mkdir -p .claude
if [ -L .claude/skills ]; then
    rm .claude/skills
fi
ln -s ../.agents/skills .claude/skills
echo "  .claude/skills -> .agents/skills"

# Codex
mkdir -p .codex
if [ -L .codex/skills ]; then
    rm .codex/skills
fi
ln -s ../.agents/skills .codex/skills
echo "  .codex/skills -> .agents/skills"

# CLAUDE.md symlink
if [ -L CLAUDE.md ] || [ -f CLAUDE.md ]; then
    rm CLAUDE.md
fi
ln -s AGENTS.md CLAUDE.md
echo "  CLAUDE.md -> AGENTS.md"

echo "Done."
