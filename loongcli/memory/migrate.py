from __future__ import annotations

import json
import logging
from pathlib import Path

from loongcli.memory.markdown_store import (
    MarkdownMemoryStore,
    _parse_frontmatter,
    _render_frontmatter,
)

logger = logging.getLogger(__name__)


def migrate_kv_to_markdown(base_dir: Path) -> int:
    """Migrate legacy kv.json entries to individual Markdown memory files.

    Returns the number of entries migrated.
    """
    kv_path = base_dir / "kv.json"
    if not kv_path.exists():
        return 0

    raw = json.loads(kv_path.read_text(encoding="utf-8"))
    store = MarkdownMemoryStore(base_dir=base_dir)
    count = 0

    for category, entries in raw.items():
        for key, val in entries.items():
            name = f"{category}-{key}"

            # Skip entries that already exist as .md files (idempotent)
            if store.load(name) is not None:
                continue

            if isinstance(val, dict) and "value" in val:
                content = val["value"]
                mem_type = val.get("type", "project")
                created = val.get("created_at")
                updated = val.get("updated_at")
            else:
                content = str(val)
                mem_type = "project"
                created = None
                updated = None

            saved_name = store.save(
                name=name,
                description=f"Migrated from {category}/{key}",
                type=mem_type,
                content=content,
            )

            # Patch timestamps if available from old data
            if created or updated:
                path = store._file_path(saved_name)
                text = path.read_text(encoding="utf-8")
                meta, body = _parse_frontmatter(text)
                if created:
                    meta["created_at"] = created
                if updated:
                    meta["updated_at"] = updated
                path.write_text(
                    _render_frontmatter(meta) + "\n\n" + body, encoding="utf-8"
                )

            count += 1

    if count > 0:
        store._rebuild_index()
        kv_path.rename(base_dir / "kv.json.bak")
        logger.info("Migrated %d memories from kv.json to markdown", count)

    return count
