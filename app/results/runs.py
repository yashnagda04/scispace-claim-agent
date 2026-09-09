"""One directory and a small metadata record per invocation."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def update_run(directory, **changes):
    path = Path(directory) / "run.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    record.update(changes, updated_at=timestamp())
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return record


def create_run(root, settings, command):
    directory = Path(root) / str(uuid4())
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "extraction").mkdir()
    (directory / "verification").mkdir()
    update_run(
        directory, run_id=directory.name, command=command, status="running",
        created_at=timestamp(), settings=settings.public_dict(),
    )
    return directory
