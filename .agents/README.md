# AI Tools Configuration

**This directory (`.agents/`) is the source of truth for all AI agent configurations.**

All AI tool configurations (`.claude/`, `.codex/`) are automatically generated or symlinked from here using the sync script.

## Directory Structure

```
.agents/
├── skills/          # Modular skills applied automatically by context
├── settings.json    # Permissions and configuration
└── README.md        # This file
```

## Quick Reference

### Edit Here (Source of Truth)

**All changes should be made in `.agents/`**:

- Add/edit skills in `.agents/skills/`
- Edit settings in `.agents/settings.json`

### Sync to AI Tools

Run the sync script to propagate changes:

```bash
bash scripts/sync-ai-tools.sh
```

This syncs to:

| Tool   | Location   | Method   |
|--------|-----------|----------|
| Claude | `.claude/` | Symlinks |
| Codex  | `.codex/`  | Symlinks |

**Important**: Never edit `.claude/` or `.codex/` directly — changes will be overwritten on next sync.
