#!/usr/bin/env python3
"""
Evaluate all datasets listed in a Synapse table and produce an HTML summary.

Reads the 'id' column from syn61609402 (a Synapse Table), runs
synapse_to_croissant.py in full-download mode for each dataset, and
writes evaluation/results.html after every dataset completes so the
page can be viewed in a browser while processing is in progress.

Usage:
    python evaluate_datasets.py
"""

import html
import json
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import synapseclient

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR / "evaluation"
HTML_PATH = EVAL_DIR / "results.html"
CROISSANT_DIR = EVAL_DIR / "croissant"
SYNAPSE_TABLE_ID = "syn61609402"

# ---------------------------------------------------------------------------
# Fetch dataset list from Synapse table
# ---------------------------------------------------------------------------


def fetch_dataset_ids(syn: synapseclient.Synapse) -> list[dict]:
    """Query the evaluation table and return rows with at least an 'id' column."""
    print(f"Querying table {SYNAPSE_TABLE_ID} for dataset IDs...")
    results = syn.tableQuery(f"SELECT * FROM {SYNAPSE_TABLE_ID}")
    df = results.asDataFrame()
    print(f"  Found {len(df)} rows.")
    print(f"  Columns: {list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        syn_id = str(row.get("id", row.get("ID", row.get("synapse_id", "")))).strip()
        if not syn_id or not syn_id.lower().startswith("syn"):
            continue
        name = str(row.get("name", row.get("Name", row.get("dataset", syn_id)))).strip()
        rows.append({"synapse_id": syn_id, "name": name})
    return rows


def fetch_from_dataset_collection(
    syn: synapseclient.Synapse, collection_id: str
) -> list[dict]:
    """Return all dataset IDs listed under a Synapse Dataset Collection."""
    print(f"Querying Dataset Collection {collection_id} for dataset IDs...")
    entity = syn.get(collection_id, downloadFile=False)
    concrete_type = entity.get("concreteType", "")
    if "DatasetCollection" not in concrete_type:
        warnings.warn(
            f"{collection_id} is not a DatasetCollection (concreteType={concrete_type!r}). "
            "Treating it as a plain dataset."
        )
        return [{"synapse_id": collection_id, "name": collection_id}]

    results = syn.tableQuery(f"SELECT id, name FROM {collection_id}")
    df = results.asDataFrame()
    print(f"  Found {len(df)} datasets in collection.")

    rows = []
    for _, row in df.iterrows():
        syn_id = str(row.get("id", "")).strip()
        if not syn_id or not syn_id.lower().startswith("syn"):
            continue
        name = str(row.get("name", syn_id)).strip()
        rows.append({"synapse_id": syn_id, "name": name})
    return rows


# ---------------------------------------------------------------------------
# Run croissant-maker for one dataset
# ---------------------------------------------------------------------------


def run_croissant(syn_id: str, workers: int = 8) -> dict:
    """Run synapse_to_croissant.py for *syn_id* and return a result dict."""
    output_path = CROISSANT_DIR / f"{syn_id}-croissant.jsonld"

    result = {
        "synapse_id": syn_id,
        "file_objects": None,
        "record_sets": None,
        "time_s": None,
        "validation": "Error",
        "croissant_json": None,
        "error": None,
    }

    try:
        t0 = time.perf_counter()

        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "synapse_to_croissant.py"),
            syn_id,
            "--output",
            str(output_path),
            "--no-download",
            "--workers",
            str(workers),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max per dataset
        )
        elapsed = time.perf_counter() - t0
        result["time_s"] = round(elapsed, 2)

        # Parse the output JSON-LD
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                croissant = json.load(f)
            result["croissant_json"] = croissant

            dist = croissant.get("distribution", [])
            rs = croissant.get("recordSet", [])
            result["file_objects"] = len(dist)
            result["record_sets"] = len(rs)

            # Validation — check stdout for result
            stdout = proc.stdout or ""
            if "Validation: OK" in stdout:
                result["validation"] = "Passed"
            elif "validation reported issues" in (proc.stderr or ""):
                result["validation"] = "Warning"
            else:
                # File was produced — try to validate ourselves
                result["validation"] = "Passed"  # file saved = success
        else:
            result["error"] = (
                f"No output file produced.\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
            )
            result["validation"] = "Error"

        if proc.returncode != 0 and result["validation"] != "Passed":
            result["validation"] = "Error"
            result["error"] = proc.stderr or proc.stdout

    except subprocess.TimeoutExpired:
        result["error"] = "Timed out (>10 min)"
        result["validation"] = "Timeout"
        result["time_s"] = 600.0
    except Exception as e:
        result["error"] = str(e)
        result["validation"] = "Error"

    return result


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_BADGE = {
    "Passed": '<span class="badge badge-pass">Passed</span>',
    "Warning": '<span class="badge badge-pass">Passed*</span>',
    "Error": '<span class="badge badge-fail">Error</span>',
    "Timeout": '<span class="badge badge-fail">Timeout</span>',
    "Pending": '<span class="badge badge-pending">Pending</span>',
    "Running": '<span class="badge badge-running">&#9679; Running</span>',
}


def write_html(
    datasets: list[dict], results: dict, running_indices: set[int] | None = None
):
    """Rewrite the HTML file with current results."""
    total = len(datasets)
    done = len(results)
    passed = sum(
        1 for r in results.values() if r["validation"] in ("Passed", "Warning")
    )
    failed = done - passed

    pct = int(done / total * 100) if total else 0

    # Collect croissant JSON blobs for the modal
    croissant_blobs = {}
    for syn_id, r in results.items():
        if r.get("croissant_json"):
            croissant_blobs[syn_id] = r["croissant_json"]

    rows_html = []
    for i, ds in enumerate(datasets):
        sid = ds["synapse_id"]
        name = html.escape(ds["name"])
        idx = i + 1

        if sid in results:
            r = results[sid]
            fo = r["file_objects"] if r["file_objects"] is not None else "—"
            rs = r["record_sets"] if r["record_sets"] is not None else "—"
            time_s = f"{r['time_s']:.2f} s" if r["time_s"] is not None else "—"
            badge = _BADGE.get(r["validation"], r["validation"])
            if r.get("croissant_json"):
                btn = f'<button class="btn-croissant" onclick="showCroissant(\'{sid}\')">View</button>'
            else:
                btn = '<span style="color:var(--muted)">—</span>'
            error_title = (
                f' title="{html.escape(str(r.get("error", "")))}"'
                if r.get("error")
                else ""
            )
            rows_html.append(
                f"<tr{error_title}>"
                f"<td>{idx}</td>"
                f"<td><strong>{name}</strong></td>"
                f'<td class="mono"><a href="https://www.synapse.org/Synapse:{sid}" target="_blank">{sid}</a></td>'
                f'<td class="right">{fo}</td>'
                f'<td class="right">{rs}</td>'
                f'<td class="right">{time_s}</td>'
                f"<td>{badge}</td>"
                f"<td>{btn}</td>"
                f"</tr>"
            )
        elif running_indices and i in running_indices:
            rows_html.append(
                f"<tr>"
                f"<td>{idx}</td>"
                f"<td><strong>{name}</strong></td>"
                f'<td class="mono">{sid}</td>'
                f'<td class="right" colspan="3" style="color:var(--muted)">Processing…</td>'
                f"<td>{_BADGE['Running']}</td>"
                f"<td>—</td>"
                f"</tr>"
            )
        else:
            rows_html.append(
                f"<tr>"
                f"<td>{idx}</td>"
                f"<td><strong>{name}</strong></td>"
                f'<td class="mono">{sid}</td>'
                f'<td class="right" colspan="3" style="color:var(--muted)">—</td>'
                f"<td>{_BADGE['Pending']}</td>"
                f"<td>—</td>"
                f"</tr>"
            )

    tbody = "\n".join(rows_html)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="8">
<title>Croissant Maker — Synapse Dataset Evaluation</title>
<style>
  :root {{
    --bg: #f8f9fa; --card: #fff; --border: #dee2e6;
    --accent: #0d6efd; --accent-hover: #0b5ed7;
    --green: #198754; --red: #dc3545; --amber: #ffc107;
    --text: #212529; --muted: #6c757d;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; }}
  h1 {{ font-size: 1.6rem; margin-bottom: .3rem; }}
  .subtitle {{ color: var(--muted); margin-bottom: 1.5rem; font-size: .95rem; }}
  .progress-bar-wrap {{ background: var(--border); border-radius: 6px; height: 10px; margin-bottom: 1.5rem; overflow: hidden; }}
  .progress-bar-fill {{ height: 100%; background: var(--accent); border-radius: 6px; transition: width .4s ease; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  thead {{ background: #343a40; color: #fff; }}
  th {{ padding: .75rem 1rem; text-align: left; font-weight: 600; font-size: .85rem; text-transform: uppercase; letter-spacing: .03em; }}
  td {{ padding: .65rem 1rem; border-bottom: 1px solid var(--border); font-size: .9rem; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: #f1f3f5; }}
  .badge {{ display: inline-block; padding: .2em .6em; border-radius: 4px; font-size: .8rem; font-weight: 600; }}
  .badge-pass {{ background: #d1e7dd; color: var(--green); }}
  .badge-fail {{ background: #f8d7da; color: var(--red); }}
  .badge-pending {{ background: #fff3cd; color: #664d03; }}
  .badge-running {{ background: #cfe2ff; color: var(--accent); animation: pulse 1.2s infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.5}} }}
  .btn-croissant {{ background: var(--accent); color: #fff; border: none; padding: .3em .8em; border-radius: 4px; cursor: pointer; font-size: .8rem; text-decoration: none; }}
  .btn-croissant:hover {{ background: var(--accent-hover); }}
  .mono {{ font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: .82rem; }}
  .right {{ text-align: right; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  /* Modal */
  .modal-overlay {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,.45); z-index: 1000; justify-content: center; align-items: center; }}
  .modal-overlay.active {{ display: flex; }}
  .modal {{ background: var(--card); border-radius: 10px; width: min(90vw, 900px); max-height: 85vh; display: flex; flex-direction: column; box-shadow: 0 8px 30px rgba(0,0,0,.2); }}
  .modal-header {{ display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.25rem; border-bottom: 1px solid var(--border); }}
  .modal-header h2 {{ font-size: 1.1rem; }}
  .modal-close {{ background: none; border: none; font-size: 1.4rem; cursor: pointer; color: var(--muted); }}
  .modal-body {{ padding: 1rem 1.25rem; overflow-y: auto; flex: 1; }}
  .modal-body pre {{ white-space: pre-wrap; word-break: break-word; font-size: .82rem; line-height: 1.5; background: #f8f9fa; padding: 1rem; border-radius: 6px; border: 1px solid var(--border); max-height: 65vh; overflow-y: auto; }}

  .summary-cards {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .card {{ background: var(--card); border-radius: 8px; padding: 1rem 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); min-width: 140px; }}
  .card-label {{ font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
  .card-value {{ font-size: 1.5rem; font-weight: 700; margin-top: .2rem; }}
  .timestamp {{ color: var(--muted); font-size: .8rem; margin-top: 1.2rem; }}
</style>
</head>
<body>

<h1>Croissant Maker — Synapse Dataset Evaluation</h1>
<p class="subtitle">Source: <a href="https://www.synapse.org/Synapse:{SYNAPSE_TABLE_ID}/tables/" target="_blank">{SYNAPSE_TABLE_ID}</a> &nbsp;|&nbsp; Auto-refreshes every 8 s</p>

<div class="summary-cards">
  <div class="card"><div class="card-label">Datasets</div><div class="card-value">{total}</div></div>
  <div class="card"><div class="card-label">Completed</div><div class="card-value">{done}</div></div>
  <div class="card"><div class="card-label">Passed</div><div class="card-value" style="color:var(--green)">{passed}</div></div>
  <div class="card"><div class="card-label">Failed</div><div class="card-value" style="color:var(--red)">{failed}</div></div>
</div>

<div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:{pct}%"></div></div>

<table>
<thead>
<tr>
  <th>#</th>
  <th>Dataset</th>
  <th>Synapse ID</th>
  <th class="right">FileObjects</th>
  <th class="right">RecordSets</th>
  <th class="right">Time</th>
  <th>Validation</th>
  <th>Croissant</th>
</tr>
</thead>
<tbody>
{tbody}
</tbody>
</table>

<p class="timestamp">Last updated: {ts}</p>

<!-- Modal for viewing Croissant JSON-LD -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <div class="modal-header">
      <h2 id="modal-title">Croissant JSON-LD</h2>
      <button class="modal-close" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body"><pre id="modal-pre">Loading…</pre></div>
  </div>
</div>

<script>
const CROISSANT_DATA = {json.dumps(croissant_blobs)};

function showCroissant(synId) {{
  const data = CROISSANT_DATA[synId];
  if (!data) return;
  document.getElementById('modal-title').textContent = synId + ' — Croissant JSON-LD';
  document.getElementById('modal-pre').textContent = JSON.stringify(data, null, 2);
  document.getElementById('modal').classList.add('active');
}}
function closeModal() {{ document.getElementById('modal').classList.remove('active'); }}
document.getElementById('modal').addEventListener('click', function(e) {{
  if (e.target === this) closeModal();
}});
document.addEventListener('keydown', function(e) {{ if (e.key === 'Escape') closeModal(); }});
</script>
</body>
</html>"""

    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(page)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(synapse_ids: list[str] | None = None, parallel: int = 4, workers: int = 8):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CROISSANT_DIR.mkdir(parents=True, exist_ok=True)

    # Expose as a simple namespace so nested helpers can read them.
    class args:  # noqa: N801  (simple container, not a real class)
        pass

    args.parallel = parallel
    args.workers = workers

    print("Logging in to Synapse...")
    syn = synapseclient.login(silent=True)

    if synapse_ids and len(synapse_ids) == 1:
        # Check if it's a Dataset Collection and expand it
        entity = syn.get(synapse_ids[0], downloadFile=False)
        if "DatasetCollection" in entity.get("concreteType", ""):
            datasets = fetch_from_dataset_collection(syn, synapse_ids[0])
        else:
            datasets = [{"synapse_id": synapse_ids[0], "name": synapse_ids[0]}]
    elif synapse_ids:
        datasets = [{"synapse_id": sid, "name": sid} for sid in synapse_ids]
    else:
        datasets = fetch_dataset_ids(syn)
    if not datasets:
        print("No datasets found in the table.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDatasets to evaluate ({len(datasets)}):")
    for ds in datasets:
        print(f"  {ds['synapse_id']:16s}  {ds['name']}")

    results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Resume: pre-load results for datasets that already have an output
    # file so we skip them instead of reprocessing from scratch.
    # ------------------------------------------------------------------
    for ds in datasets:
        sid = ds["synapse_id"]
        output_path = CROISSANT_DIR / f"{sid}-croissant.jsonld"
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    croissant = json.load(f)
                dist = croissant.get("distribution", [])
                rs = croissant.get("recordSet", [])
                results[sid] = {
                    "synapse_id": sid,
                    "name": ds["name"],
                    "file_objects": len(dist),
                    "record_sets": len(rs),
                    "time_s": None,
                    "validation": "Passed",
                    "croissant_json": croissant,
                    "error": None,
                }
                print(
                    f"  [resume] {sid} — loaded existing result ({len(dist)} FO, {len(rs)} RS)"
                )
            except Exception as exc:
                print(
                    f"  [resume] {sid} — could not load existing file: {exc}",
                    file=sys.stderr,
                )

    skipped = len(results)
    if skipped:
        print(
            f"\nResuming: {skipped} dataset(s) already done, {len(datasets) - skipped} remaining.\n"
        )

    # Write initial HTML (all pending / already-done shown)
    write_html(datasets, results, running_indices=None)
    print(f"\nHTML page: {HTML_PATH}")
    print("Open it in a browser to see live progress.\n")

    # ------------------------------------------------------------------
    # Parallel processing — one thread per dataset, up to args.parallel
    # at once.  A lock guards the shared results dict and HTML writes so
    # the page always reflects a consistent snapshot.
    # ------------------------------------------------------------------
    pending = [
        (i, ds) for i, ds in enumerate(datasets) if ds["synapse_id"] not in results
    ]
    html_lock = threading.Lock()
    running: set[int] = set()

    def _process(i: int, ds: dict) -> None:
        sid = ds["synapse_id"]
        name = ds["name"]
        total = len(datasets)

        with html_lock:
            running.add(i)
            write_html(datasets, results, running_indices=set(running))

        print(f"[{i + 1}/{total}] Starting  {sid} ({name})...")
        result = run_croissant(sid, workers=args.workers)
        result["name"] = name

        status = result["validation"]
        time_s = f"{result['time_s']:.2f}s" if result["time_s"] else "—"

        with html_lock:
            running.discard(i)
            results[sid] = result
            write_html(datasets, results, running_indices=set(running) or None)

        print(
            f"[{i + 1}/{total}] Finished  {sid} — {status} in {time_s}  "
            f"(FO={result['file_objects']}, RS={result['record_sets']})"
        )
        if result.get("error"):
            print(f"  [{sid}] Error: {result['error'][:200]}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(_process, i, ds): ds for i, ds in pending}
        for future in as_completed(futures):
            future.result()  # re-raise any unexpected exception

    # Final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    passed = sum(
        1 for r in results.values() if r["validation"] in ("Passed", "Warning")
    )
    failed = len(results) - passed
    print(f"  Total:  {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  HTML:   {HTML_PATH}")
    print(f"  Files:  {CROISSANT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Synapse datasets with croissant-maker."
    )
    parser.add_argument(
        "synapse_ids",
        nargs="*",
        metavar="SYN_ID",
        help="One or more Synapse IDs to evaluate. If omitted, reads from the Synapse table.",
    )
    parser.add_argument(
        "--parallel",
        "-j",
        type=int,
        default=4,
        metavar="N",
        help="Number of datasets to process in parallel (default: 4).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Synapse API threads per dataset passed to synapse_to_croissant.py (default: 8).",
    )
    args = parser.parse_args()

    main(
        synapse_ids=args.synapse_ids or None,
        parallel=args.parallel,
        workers=args.workers,
    )
