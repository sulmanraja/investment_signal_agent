"""Long-term memory — persists run summaries across analysis sessions.

Stores structured run records in run_store.json and provides
retrieval by ticker or date range.
"""

import json
import os
from datetime import datetime
from pathlib import Path

RUN_STORE_PATH = Path(__file__).parent / "run_store.json"


class LongTermMemory:
    """JSON-backed store for persisting analysis run summaries."""

    def __init__(self, store_path: Path = RUN_STORE_PATH):
        self.store_path = store_path
        self._records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.store_path.exists():
            try:
                return json.loads(self.store_path.read_text())
            except json.JSONDecodeError:
                return []
        return []

    def _save(self) -> None:
        self.store_path.write_text(json.dumps(self._records, indent=2))

    def store_run(
        self,
        ticker: str,
        signal_score: float,
        stance: str,
        recommendation: str,
        summary: str,
    ) -> dict:
        """Persist one analysis run.

        Returns the stored record dict.
        """
        record = {
            "id": len(self._records) + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "ticker": ticker.upper(),
            "signal_score": signal_score,
            "stance": stance,
            "recommendation": recommendation,
            "summary": summary,
        }
        self._records.append(record)
        self._save()
        return record

    def get_by_ticker(self, ticker: str) -> list[dict]:
        return [r for r in self._records if r["ticker"] == ticker.upper()]

    def get_latest(self, n: int = 5) -> list[dict]:
        return self._records[-n:]

    def clear(self) -> None:
        self._records = []
        self._save()

    def __len__(self) -> int:
        return len(self._records)


if __name__ == "__main__":
    print("=== long_term memory local test ===")
    mem = LongTermMemory()
    rec = mem.store_run(
        ticker="NVDA",
        signal_score=82.0,
        stance="Bullish",
        recommendation="BUY",
        summary="Record AI chip demand with Blackwell on schedule.",
    )
    print(f"Stored run #{rec['id']} for {rec['ticker']}")
    print(f"Total records: {len(mem)}")
    for r in mem.get_by_ticker("NVDA"):
        print(f"  [{r['timestamp']}] {r['recommendation']} score={r['signal_score']}")
