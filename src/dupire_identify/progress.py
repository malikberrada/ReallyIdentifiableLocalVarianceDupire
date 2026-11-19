"""Count real work; heartbeat opaque tasks without inventing percentages."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time


class Progress:
    def __init__(self, mode="plain", heartbeat=2.0, log=None):
        self.mode = mode
        self.heartbeat = max(.1, heartbeat)
        self.log = Path(log) if log else None
        self.started = time.monotonic()
        self.lock = threading.RLock()
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)

    def event(self, stage, status="running", done=None, total=None, **extra):
        row = dict(time=datetime.now(timezone.utc).isoformat(),
                   elapsed_s=round(time.monotonic()-self.started, 3),
                   stage=stage, status=status, **extra)
        if done is not None:
            row["done"] = done
        if total is not None:
            row["total"] = total
        with self.lock:
            if self.log:
                with self.log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False)+"\n")
            if self.mode == "off":
                return
            if self.mode == "json":
                print(json.dumps(row, ensure_ascii=False), file=sys.stderr, flush=True)
            else:
                count = ""
                if total:
                    ratio = min(1, max(0, (done or 0)/total))
                    count = f" [{'#'*int(18*ratio):18}] {done or 0}/{total} ({100*ratio:.0f}%)"
                detail = " ".join(f"{k}={v}" for k, v in extra.items())
                print(f"[{row['elapsed_s']:8.1f}s] {stage}: {status}{count} {detail}",
                      file=sys.stderr, flush=True)

    @contextmanager
    def phase(self, stage, **extra):
        self.event(stage, "start", **extra)
        stop = threading.Event()
        def beat():
            while not stop.wait(self.heartbeat):
                self.event(stage, "working", **extra)
        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        t = time.monotonic()
        try:
            yield
        except BaseException as exc:
            self.event(stage, "failed", error=str(exc))
            raise
        else:
            self.event(stage, "done", duration_s=round(time.monotonic()-t, 3))
        finally:
            stop.set()
            thread.join(timeout=self.heartbeat+.1)


QUIET = Progress(mode="off")
