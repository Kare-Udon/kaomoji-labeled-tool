from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from threading import Event
from threading import Lock
from threading import Thread


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/step6/submission_strict_gpt-5.4_v4/submit_batches.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("step6_submit_batches", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for idx in range(count):
            fh.write(json.dumps({"custom_id": f"id-{idx}"}, ensure_ascii=False) + "\n")


def test_split_jsonl_by_request_count_splits_into_300_request_parts(tmp_path):
    module = load_module()

    source = tmp_path / "full.jsonl"
    parts_dir = tmp_path / "parts"
    write_jsonl(source, 650)

    stats = module.split_jsonl_by_request_count(source, parts_dir, max_requests_per_part=300)

    assert [item["line_count"] for item in stats] == [300, 300, 50]
    assert [Path(item["path"]).name for item in stats] == [
        "part-001.jsonl",
        "part-002.jsonl",
        "part-003.jsonl",
    ]


def test_sequential_runner_processes_next_part_after_success(tmp_path):
    module = load_module()

    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    write_jsonl(parts_dir / "part-001.jsonl", 2)
    write_jsonl(parts_dir / "part-002.jsonl", 2)

    notifications = []

    class FakeOpenAI:
        def __init__(self):
            self.created = []
            self.downloaded = []

        def upload_file(self, file_path: Path) -> str:
            return f"file-{file_path.stem}"

        def create_batch(self, input_file_id: str, endpoint: str, completion_window: str, metadata=None) -> dict:
            self.created.append(input_file_id)
            return {"id": f"batch-{input_file_id}", "status": "validating"}

        def retrieve_batch(self, batch_id: str) -> dict:
            return {
                "id": batch_id,
                "status": "completed",
                "output_file_id": f"out-{batch_id}",
                "error_file_id": None,
            }

        def download_file(self, file_id: str, output_path: Path) -> None:
            output_path.write_text(file_id, encoding="utf-8")
            self.downloaded.append(file_id)

    runner = module.SequentialBatchRunner(
        parts_dir=parts_dir,
        output_dir=tmp_path / "results",
        state_path=tmp_path / "state.json",
        api_client=FakeOpenAI(),
        poll_interval_seconds=0,
        notifier=lambda text: notifications.append(text),
    )

    runner.run_all()

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["parts"]["part-001.jsonl"]["status"] == "completed"
    assert state["parts"]["part-002.jsonl"]["status"] == "completed"
    assert len(notifications) == 2


def test_sequential_runner_stops_immediately_on_error(tmp_path):
    module = load_module()

    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    write_jsonl(parts_dir / "part-001.jsonl", 2)
    write_jsonl(parts_dir / "part-002.jsonl", 2)

    notifications = []

    class FakeOpenAI:
        def upload_file(self, file_path: Path) -> str:
            return "file-1"

        def create_batch(self, input_file_id: str, endpoint: str, completion_window: str, metadata=None) -> dict:
            raise RuntimeError("create failed")

        def retrieve_batch(self, batch_id: str) -> dict:
            raise AssertionError("should not poll after create error")

        def download_file(self, file_id: str, output_path: Path) -> None:
            raise AssertionError("should not download after create error")

    runner = module.SequentialBatchRunner(
        parts_dir=parts_dir,
        output_dir=tmp_path / "results",
        state_path=tmp_path / "state.json",
        api_client=FakeOpenAI(),
        poll_interval_seconds=0,
        notifier=lambda text: notifications.append(text),
    )

    try:
        runner.run_all()
    except RuntimeError as exc:
        assert str(exc) == "create failed"
    else:
        raise AssertionError("expected RuntimeError")

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["parts"]["part-001.jsonl"]["status"] == "error"
    assert state["parts"]["part-002.jsonl"]["status"] == "pending"
    assert len(notifications) == 1


def test_default_notifier_appends_url_encoded_message(monkeypatch):
    module = load_module()
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(url, timeout=0):
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    module.default_notifier("part-001 completed", "https://example.com/base/")

    assert captured["url"] == "https://example.com/base/part-001%20completed"
    assert captured["timeout"] == 30


def test_sequential_runner_does_not_fail_when_notification_times_out(tmp_path):
    module = load_module()

    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    write_jsonl(parts_dir / "part-001.jsonl", 2)

    class FakeOpenAI:
        def upload_file(self, file_path: Path) -> str:
            return f"file-{file_path.stem}"

        def create_batch(self, input_file_id: str, endpoint: str, completion_window: str, metadata=None) -> dict:
            return {"id": f"batch-{input_file_id}", "status": "validating"}

        def retrieve_batch(self, batch_id: str) -> dict:
            return {
                "id": batch_id,
                "status": "completed",
                "output_file_id": f"out-{batch_id}",
                "error_file_id": None,
            }

        def download_file(self, file_id: str, output_path: Path) -> None:
            output_path.write_text(file_id, encoding="utf-8")

    def failing_notifier(text: str) -> None:
        raise TimeoutError("notify timed out")

    runner = module.SequentialBatchRunner(
        parts_dir=parts_dir,
        output_dir=tmp_path / "results",
        state_path=tmp_path / "state.json",
        api_client=FakeOpenAI(),
        poll_interval_seconds=0,
        notifier=failing_notifier,
    )

    runner.run_all()

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["parts"]["part-001.jsonl"]["status"] == "completed"
    assert "notify timed out" in state["parts"]["part-001.jsonl"]["last_error"]


def test_sequential_runner_respects_max_parts_limit(tmp_path):
    module = load_module()

    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    write_jsonl(parts_dir / "part-001.jsonl", 2)
    write_jsonl(parts_dir / "part-002.jsonl", 2)
    write_jsonl(parts_dir / "part-003.jsonl", 2)

    class FakeOpenAI:
        def __init__(self):
            self.created = []

        def upload_file(self, file_path: Path) -> str:
            return f"file-{file_path.stem}"

        def create_batch(self, input_file_id: str, endpoint: str, completion_window: str, metadata=None) -> dict:
            self.created.append(input_file_id)
            return {"id": f"batch-{input_file_id}", "status": "validating"}

        def retrieve_batch(self, batch_id: str) -> dict:
            return {
                "id": batch_id,
                "status": "completed",
                "output_file_id": f"out-{batch_id}",
                "error_file_id": None,
            }

        def download_file(self, file_id: str, output_path: Path) -> None:
            output_path.write_text(file_id, encoding="utf-8")

    api = FakeOpenAI()
    runner = module.SequentialBatchRunner(
        parts_dir=parts_dir,
        output_dir=tmp_path / "results",
        state_path=tmp_path / "state.json",
        api_client=api,
        poll_interval_seconds=0,
        notifier=lambda text: None,
    )

    runner.run_all(max_parts=2)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["parts"]["part-001.jsonl"]["status"] == "completed"
    assert state["parts"]["part-002.jsonl"]["status"] == "completed"
    assert state["parts"]["part-003.jsonl"]["status"] == "pending"
    assert len(api.created) == 2


def test_runner_respects_concurrency_and_total_limit(tmp_path):
    module = load_module()

    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    for idx in range(1, 6):
        write_jsonl(parts_dir / f"part-{idx:03d}.jsonl", 2)

    class FakeOpenAI:
        def __init__(self):
            self.created = []
            self.active = 0
            self.max_active = 0
            self.lock = Lock()
            self.ready = Event()

        def upload_file(self, file_path: Path) -> str:
            return f"file-{file_path.stem}"

        def create_batch(self, input_file_id: str, endpoint: str, completion_window: str, metadata=None) -> dict:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.created.append(input_file_id)
                if self.active >= 2:
                    self.ready.set()
            return {"id": f"batch-{input_file_id}", "status": "validating"}

        def retrieve_batch(self, batch_id: str) -> dict:
            self.ready.wait(timeout=1)
            with self.lock:
                self.active -= 1
            return {
                "id": batch_id,
                "status": "completed",
                "output_file_id": f"out-{batch_id}",
                "error_file_id": None,
            }

        def download_file(self, file_id: str, output_path: Path) -> None:
            output_path.write_text(file_id, encoding="utf-8")

    api = FakeOpenAI()
    runner = module.SequentialBatchRunner(
        parts_dir=parts_dir,
        output_dir=tmp_path / "results",
        state_path=tmp_path / "state.json",
        api_client=api,
        poll_interval_seconds=0,
        notifier=lambda text: None,
    )

    runner.run_all(max_parts=3, max_concurrent_parts=2)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    completed = [name for name, item in state["parts"].items() if item["status"] == "completed"]
    pending = [name for name, item in state["parts"].items() if item["status"] == "pending"]
    assert len(completed) == 3
    assert len(pending) == 2
    assert len(api.created) == 3
    assert api.max_active == 2
