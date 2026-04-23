from __future__ import annotations

import csv
import json
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from cimpy.llm_routing.orchestrator import Orchestrator


USER_INPUTS: List[str] = [
    "Was war die durchschnittliche Wirkleistung von Trafo 19-20 am 2026-01-09?",
    "Was ist die Nennleistung von Trafo 19-20? Nutze historical cim.",
    "Wie war die durchschnittliche Auslastung von Trafo 19-20 am 09.01.2026? ",
    "Was sind die direkten Nachbarn von Last 3? Nutze historical cim.",
]

OUTPUT_DIR = Path("router_batch_results") / datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class SummaryRow:
    run_id: int
    user_input: str
    route: str
    answer: str
    status: str
    error: str
    details: str
    duration_seconds: float
    json_path: str


def json_default(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return "<non-serializable>"


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)


def save_csv(path: Path, rows: List[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "user_input",
                "route",
                "answer",
                "status",
                "error",
                "details",
                "duration_seconds",
                "json_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def extract_answer(out: Dict[str, Any]) -> str:
    if not isinstance(out, dict):
        return str(out)

    if out.get("route") == "ASK_USER":
        return str(out.get("question", ""))

    result = out.get("result", {})
    if isinstance(result, dict):
        return str(result.get("answer", ""))
    return ""


def extract_status(out: Dict[str, Any]) -> str:
    result = out.get("result", {})
    if isinstance(result, dict):
        return str(result.get("status", ""))
    return ""


def extract_error(out: Dict[str, Any]) -> str:
    result = out.get("result", {})
    if isinstance(result, dict):
        return str(result.get("error", ""))
    return ""


def extract_details(out: Dict[str, Any]) -> str:
    result = out.get("result", {})
    if isinstance(result, dict):
        return str(result.get("details", ""))
    return ""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    orch = Orchestrator()
    rows: List[SummaryRow] = []

    metadata = {
        "started_at": datetime.now().isoformat(),
        "entrypoint": "cimpy.llm_routing.orchestrator.Orchestrator.handle",
        "equivalent_cli": r"python -m cimpy.llm_routing.run_router",
        "user_inputs": USER_INPUTS,
    }
    save_json(OUTPUT_DIR / "run_metadata.json", metadata)

    total_start = time.perf_counter()

    for idx, user_input in enumerate(USER_INPUTS, start=1):
        print(f"\n=== RUN {idx}/{len(USER_INPUTS)} ===")
        print("INPUT:", user_input)

        output_path = OUTPUT_DIR / f"{idx:03d}_router_result.json"
        run_start = time.perf_counter()

        try:
            out = orch.handle(user_input)
            duration_seconds = round(time.perf_counter() - run_start, 3)

            payload = {
                "run_id": idx,
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "duration_seconds": duration_seconds,
                "router_output": out,
            }
            save_json(output_path, payload)

            route = str(out.get("route", "")) if isinstance(out, dict) else ""
            answer = extract_answer(out)
            status = extract_status(out)
            error = extract_error(out)
            details = extract_details(out)

            rows.append(
                SummaryRow(
                    run_id=idx,
                    user_input=user_input,
                    route=route,
                    answer=answer,
                    status=status,
                    error=error,
                    details=details,
                    duration_seconds=duration_seconds,
                    json_path=str(output_path),
                )
            )

            print("ROUTE:", route)
            print("DAUER (s):", duration_seconds)
            print("ANSWER:", answer)

        except Exception as e:
            duration_seconds = round(time.perf_counter() - run_start, 3)

            payload = {
                "run_id": idx,
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "duration_seconds": duration_seconds,
                "router_output": {
                    "route": "ERROR",
                    "result": {
                        "status": "error",
                        "error": type(e).__name__,
                        "details": str(e),
                        "traceback": traceback.format_exc(),
                        "answer": f"Batch-Ausführung fehlgeschlagen: {type(e).__name__}: {e}",
                    },
                },
            }
            save_json(output_path, payload)

            rows.append(
                SummaryRow(
                    run_id=idx,
                    user_input=user_input,
                    route="ERROR",
                    answer=payload["router_output"]["result"]["answer"],
                    status="error",
                    error=type(e).__name__,
                    details=str(e),
                    duration_seconds=duration_seconds,
                    json_path=str(output_path),
                )
            )

            print("ROUTE: ERROR")
            print("DAUER (s):", duration_seconds)
            print("ERROR:", type(e).__name__, str(e))

    save_csv(OUTPUT_DIR / "summary.csv", rows)

    total_duration_seconds = round(time.perf_counter() - total_start, 3)

    finished_metadata = {
        **metadata,
        "finished_at": datetime.now().isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "num_runs": len(rows),
        "num_errors": sum(1 for row in rows if row.route == "ERROR" or row.status == "error"),
        "total_duration_seconds": total_duration_seconds,
        "average_duration_seconds": round(
            sum(row.duration_seconds for row in rows) / len(rows), 3
        ) if rows else 0.0,
    }
    save_json(OUTPUT_DIR / "run_metadata.json", finished_metadata)

    print("\nFertig.")
    print("Gesamtdauer (s):", total_duration_seconds)
    print("Ergebnisse in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()