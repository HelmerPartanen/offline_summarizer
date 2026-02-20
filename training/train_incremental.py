import os
import sys
import json
import argparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))

from prepare_feedback import prepare_data
from train_dpo import train_dpo


def trigger_backend_reload(reload_url="http://localhost:8000/reload_model"):
    payload = json.dumps({}).encode("utf-8")
    request = Request(
        reload_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
            return {"ok": True, "response": body}
    except HTTPError as exc:
        return {"ok": False, "reason": f"HTTP {exc.code}: {exc.reason}"}
    except URLError as exc:
        return {"ok": False, "reason": f"Backend not reachable: {exc.reason}"}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def run_incremental_training(reload_backend=True, fast_mode=True, max_new_pairs=8):
    print("=== Incremental Training Pipeline (CPU-friendly) ===")
    print(f"Mode: {'fast' if fast_mode else 'quality'} | Max new pairs this run: {max_new_pairs if max_new_pairs else 'all'}")
    prep = prepare_data()

    print(
        f"Feedback rows: {prep['total_feedback_rows']} | "
        f"Usable pairs: {prep['usable_pairs']} | "
        f"Unique pairs: {prep['unique_pairs']} | "
        f"New pairs: {prep['new_pairs']}"
    )

    if prep["new_pairs"] == 0:
        print("No new corrected feedback found. Nothing to train.")
        return {
            "trained": False,
            "reason": "no_new_pairs",
            "prep": prep,
            "reload": None
        }

    capped_pairs = max_new_pairs if (fast_mode and max_new_pairs and max_new_pairs > 0) else None
    result = train_dpo(
        data_file=prep["incremental_output_file"],
        fast_mode=fast_mode,
        max_train_samples=capped_pairs
    )

    reload_result = None
    if reload_backend and result.get("trained"):
        reload_result = trigger_backend_reload()
        if reload_result.get("ok"):
            print("Backend model reload succeeded.")
        else:
            print(f"Backend reload skipped/failed: {reload_result.get('reason')}")

    return {
        "trained": result.get("trained", False),
        "training": result,
        "prep": prep,
        "reload": reload_result
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run incremental DPO training from collected feedback.")
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Use slower quality mode (trains on all pending pairs with larger sequence lengths)."
    )
    parser.add_argument(
        "--max-new-pairs",
        type=int,
        default=8,
        help="In fast mode, train at most this many new feedback pairs per run (default: 8)."
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Skip backend model reload call after training."
    )
    args = parser.parse_args()

    summary = run_incremental_training(
        reload_backend=not args.no_reload,
        fast_mode=not args.quality,
        max_new_pairs=args.max_new_pairs
    )
    print("=== Pipeline summary ===")
    print(json.dumps(summary, indent=2))
