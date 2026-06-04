"""HTTP server for the reconcile_tiers viewer.

Serves static files from the workspace root (so `/reconcile_tiers/web/*`,
`/pipeline-outputs/<uuid>/*`, etc. all resolve) and exposes the small JSON
APIs the viewer calls into:

    GET  /flag-queues?uuid=<uuid>   →  most recent auto-scan queue + diff
    POST /flag-queue                →  persist a viewer-sent queue + fold
                                       dismissed items into the calibration
    POST /context-action            →  dispatch into
                                       reconcile_tiers.quick_actions.REGISTRY
                                       (right-click menu actions; pure)
    POST /gemini/chat               →  free-form chat → dev_tools.REGISTRY via
                                       Gemini function calling
                                       (gated by GEMINI_API_KEY)
    GET  /jobs/<id>                 →  status + log_tail of a dev_tools job
    GET  /room-postprocessing/graph →  corner-sharing element graph JSON
                                       (?uuid=, optional corner_tol=)

Run:

    python -m reconcile_tiers.server                # default port 8080
    VIEWER_PORT=8765 python -m reconcile_tiers.server

The legacy `reconcile/viewer_server.py` (now archived) had a much larger surface
area — orthophoto tiles, V3 roof proposals, calibration UI, etc. — none of
which the V2 tier viewer uses. This module is intentionally small.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
import urllib.parse
import uuid as _uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Workspace root = parent of this package. Used as the static-file root.
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
FLAG_QUEUES_ROOT = WORKSPACE_ROOT / ".context" / "flag-queues"
FLAG_CALIBRATION_ROOT = WORKSPACE_ROOT / ".context" / "flag-calibration"
ROOF_RATINGS_PATH = WORKSPACE_ROOT / ".context" / "roof_ratings.json"

HOST = os.environ.get("VIEWER_HOST", "127.0.0.1")
PORT = int(
    os.environ.get("VIEWER_PORT")
    or os.environ.get("CONDUCTOR_PORT")
    or os.environ.get("PORT")
    or "8080"
)

VIEWER_PATH = "/reconcile_tiers/web/viewer-tiers.html"
CORNER_GRAPH_VIEWER_PATH = "/reconcile_tiers/web/viewer-corner-graph.html"

# Reasons the rater can attach to a low rating to label the dominant failure
# mode. Captured optionally on rate-button click; feeds future calibration
# of the roof_quality predictor.
RATING_REASONS = {
    "fragments",
    "missing_dormer",
    "wrong_ridge",
    "splayed_wing",
    "daylight_through",
    "half_shell",
    "other",
}

# Reasons used when the rater marks a building as `upstream_error`. Lets us
# distinguish viewer-render breakage (fixable in this repo) from genuine
# extraction failures (upstream).
UPSTREAM_ERROR_REASONS = {
    "viewer_broken_render",
    "extraction_incomplete",
    "not_my_building",
    "other",
}

# ---- helpers -----------------------------------------------------------------

_SAFE_UUID_CHARS = set("0123456789abcdefABCDEF-")


def _is_safe_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if not (8 <= len(value) <= 64):
        return False
    return all(c in _SAFE_UUID_CHARS for c in value)


def _split_locator(locator: str) -> tuple[str, list[str]]:
    segments = locator.split("::", 2)
    if len(segments) != 3:
        return "", []
    kind = segments[1]
    parts = segments[2].split(":") if segments[2] else []
    return kind, parts


def _ts() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


# ---- request handler ---------------------------------------------------------


class TierServerHandler(SimpleHTTPRequestHandler):
    """Minimal viewer + flag-queue server."""

    def __init__(self, *args, directory: str | None = None, **kwargs) -> None:
        if directory is None:
            directory = str(WORKSPACE_ROOT)
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        # SimpleHTTPRequestHandler logs to stderr; keep that but compact.
        sys.stderr.write(
            f"{self.address_string()} - - [{self.log_date_time_string()}] "
            f"{format % args}\n"
        )

    def end_headers(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path.endswith((".html", ".js", ".css", ".json")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # --- routing ---------------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/viewer-tiers.html"):
            self._redirect(VIEWER_PATH)
            return
        if parsed.path == "/viewer-corner-graph.html":
            self._redirect(CORNER_GRAPH_VIEWER_PATH)
            return
        if parsed.path == "/room-postprocessing/graph":
            self._handle_room_postprocessing_graph(parsed.query)
            return
        if parsed.path == "/flag-queues":
            self._handle_flag_queue_get(parsed.query)
            return
        if parsed.path == "/roof-rating":
            self._handle_roof_rating_get()
            return
        if parsed.path.startswith("/jobs/"):
            self._handle_job_get(parsed.path[len("/jobs/") :])
            return
        if parsed.path.startswith("/labeler/"):
            self._handle_labeler_get(parsed.path, parsed.query)
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/flag-queue":
            self._handle_flag_queue_post()
            return
        if parsed.path == "/roof-rating":
            self._handle_roof_rating_post()
            return
        if parsed.path == "/context-action":
            self._handle_context_action()
            return
        if parsed.path == "/gemini/chat":
            self._handle_gemini_chat()
            return
        if parsed.path.startswith("/labeler/"):
            self._handle_labeler_post(parsed.path)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    # --- helpers ---------------------------------------------------------------

    def _redirect(self, target: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", target)
        self.end_headers()

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --- quick-action endpoints ------------------------------------------------

    def _handle_context_action(self) -> None:
        """Dispatch into reconcile_tiers.quick_actions.REGISTRY.

        Body: {"action": <name>, "params": {...}}
        Returns: the action's JSON-serializable result.
        """
        from reconcile_tiers import quick_actions

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return
        if not isinstance(body, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Body must be an object")
            return
        action = body.get("action")
        if not isinstance(action, str) or action not in quick_actions.REGISTRY:
            self.send_error(
                HTTPStatus.BAD_REQUEST,
                f"unknown action; valid: {sorted(quick_actions.REGISTRY)}",
            )
            return
        params = body.get("params") or {}
        if not isinstance(params, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "params must be an object")
            return
        try:
            result = quick_actions.dispatch(action, **params)
        except (LookupError, FileNotFoundError) as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        except (ValueError, TypeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        self._send_json(HTTPStatus.OK, {"action": action, "result": result})

    # --- jobs endpoints --------------------------------------------------------

    def _handle_job_get(self, raw_id: str) -> None:
        from reconcile_tiers import jobs as jobs_module

        job_id = raw_id.split("/", 1)[0].strip()
        if not job_id:
            self.send_error(HTTPStatus.BAD_REQUEST, "missing job id")
            return
        job = jobs_module.REGISTRY.get(job_id)
        if job is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "no such job"})
            return
        self._send_json(HTTPStatus.OK, job.to_public())

    # --- gemini chat endpoint --------------------------------------------------

    def _handle_gemini_chat(self) -> None:
        from reconcile_tiers import gemini_agent

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return
        if not isinstance(body, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Body must be an object")
            return
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self.send_error(HTTPStatus.BAD_REQUEST, "prompt is required")
            return
        targets = body.get("targets") or []
        if not isinstance(targets, list) or any(
            not isinstance(t, str) for t in targets
        ):
            self.send_error(HTTPStatus.BAD_REQUEST, "targets must be a list of strings")
            return

        try:
            result = gemini_agent.chat(prompt, targets=targets)
        except gemini_agent.GeminiUnavailable as exc:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        self._send_json(HTTPStatus.OK, result)

    # --- room postprocessing (corner graph) ------------------------------------

    def _handle_room_postprocessing_graph(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        building_uuid = (params.get("uuid") or [""])[0]
        if not _is_safe_uuid(building_uuid):
            self.send_error(HTTPStatus.BAD_REQUEST, "uuid query param required")
            return
        tol_raw = (params.get("corner_tol") or ["0.05"])[0]
        try:
            corner_tol = float(tol_raw)
        except (TypeError, ValueError):
            self.send_error(HTTPStatus.BAD_REQUEST, "corner_tol must be a number")
            return

        from reconcile_tiers.room_postprocessing.export import build_corner_graph

        payload_path = WORKSPACE_ROOT / "pipeline-outputs" / building_uuid / "tier_payload.json"
        if not payload_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "tier_payload.json not found")
            return
        try:
            payload = json.loads(payload_path.read_text())
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Invalid tier_payload.json")
            return
        self._send_json(HTTPStatus.OK, build_corner_graph(payload, corner_tol=corner_tol))

    # --- flag-queue endpoints --------------------------------------------------

    def _handle_flag_queue_get(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        building_uuid = (params.get("uuid") or [""])[0]
        if not _is_safe_uuid(building_uuid):
            self.send_error(HTTPStatus.BAD_REQUEST, "uuid query param required")
            return
        latest = FLAG_QUEUES_ROOT / building_uuid / "auto-scan-latest.json"
        if not latest.exists():
            self._send_json(HTTPStatus.OK, {"items": [], "source": None})
            return
        try:
            data = json.loads(latest.read_text())
        except Exception:
            self._send_json(HTTPStatus.OK, {"items": [], "source": None})
            return
        self._send_json(HTTPStatus.OK, data)

    def _handle_flag_queue_post(self) -> None:
        """Persist a viewer-sent queue + fold dismissed items into the calibration."""
        import base64

        from reconcile_tiers.audit import calibration as calib

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return
        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Body must be an object")
            return

        building_uuid = payload.get("building_uuid")
        if not _is_safe_uuid(building_uuid):
            self.send_error(HTTPStatus.BAD_REQUEST, "building_uuid is required")
            return

        items_in = payload.get("items")
        if not isinstance(items_in, list) or not items_in:
            self.send_error(HTTPStatus.BAD_REQUEST, "items must be a non-empty list")
            return

        cleaned: list[dict[str, Any]] = []
        for entry in items_in:
            if not isinstance(entry, dict):
                continue
            locator = entry.get("locator")
            if not isinstance(locator, str) or "::" not in locator:
                continue
            kind, parts = _split_locator(locator)
            cleaned.append(
                {
                    "id": entry.get("id") or _uuid.uuid4().hex[:12],
                    "locator": locator,
                    "kind": entry.get("kind") or kind,
                    "parts": entry.get("parts")
                    if isinstance(entry.get("parts"), list)
                    else parts,
                    "rule": entry.get("rule"),
                    "note": entry.get("note"),
                    "severity": entry.get("severity"),
                    "evidence": entry.get("evidence"),
                    "dismissed": bool(entry.get("dismissed", False)),
                }
            )
        if not cleaned:
            self.send_error(HTTPStatus.BAD_REQUEST, "no valid items in body")
            return

        timestamp = _ts()
        queue_dir = FLAG_QUEUES_ROOT / building_uuid
        queue_dir.mkdir(parents=True, exist_ok=True)
        queue_path = queue_dir / f"{timestamp}.json"

        screenshot_rel: str | None = None
        screenshot_data_url = payload.get("screenshot_data_url")
        if isinstance(screenshot_data_url, str) and screenshot_data_url.startswith(
            "data:image/"
        ):
            try:
                _, b64 = screenshot_data_url.split(",", 1)
                png_bytes = base64.b64decode(b64)
                shot_path = queue_dir / f"{timestamp}.png"
                shot_path.write_bytes(png_bytes)
                screenshot_rel = shot_path.name
            except Exception:
                screenshot_rel = None

        source = payload.get("source")
        if source not in ("viewer", "merged"):
            source = "viewer"

        queue = {
            "schema": "flag-queue/v1",
            "building_uuid": building_uuid,
            "created": timestamp,
            "source": source,
            "screenshot": screenshot_rel,
            "items": cleaned,
        }
        queue_path.write_text(json.dumps(queue, indent=2))

        # Fold dismissed auto-flags into the persistent calibration so future
        # cohort scans suppress them by default.
        calibration = calib.load(building_uuid, FLAG_CALIBRATION_ROOT)
        prior_dismiss_count = len(calibration.get("dismissals") or [])
        calib.merge_dismissals(calibration, cleaned, timestamp=timestamp)
        new_dismiss_count = len(calibration.get("dismissals") or [])
        if new_dismiss_count != prior_dismiss_count or any(
            it.get("dismissed") and it.get("rule") for it in cleaned
        ):
            calib.save(building_uuid, FLAG_CALIBRATION_ROOT, calibration)

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "queue_path": str(queue_path.relative_to(WORKSPACE_ROOT)),
                "item_count": len(cleaned),
                "dismissals_recorded": new_dismiss_count - prior_dismiss_count,
            },
        )

    # --- roof-rating endpoints -------------------------------------------------

    def _read_roof_ratings(self) -> dict[str, Any]:
        if not ROOF_RATINGS_PATH.exists():
            return {}
        try:
            data = json.loads(ROOF_RATINGS_PATH.read_text())
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_roof_ratings(self, data: dict[str, Any]) -> None:
        ROOF_RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = ROOF_RATINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(ROOF_RATINGS_PATH)

    def _handle_roof_rating_get(self) -> None:
        self._send_json(HTTPStatus.OK, self._read_roof_ratings())

    def _handle_roof_rating_post(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return
        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Body must be an object")
            return

        building_uuid = payload.get("uuid")
        if not _is_safe_uuid(building_uuid):
            self.send_error(HTTPStatus.BAD_REQUEST, "uuid is required")
            return

        rating_in = payload.get("rating")
        if isinstance(rating_in, bool):
            rating: int | str | None = None
            invalid = True
        elif isinstance(rating_in, int) and 1 <= rating_in <= 5:
            rating = rating_in
            invalid = False
        elif rating_in == "upstream_error":
            rating = "upstream_error"
            invalid = False
        elif rating_in is None:
            rating = None
            invalid = False
        else:
            rating = None
            invalid = True

        if invalid:
            self.send_error(
                HTTPStatus.BAD_REQUEST,
                "rating must be 1..5, 'upstream_error', or null",
            )
            return

        reason_in = payload.get("reason")
        if reason_in is None:
            reason: str | None = None
        elif isinstance(reason_in, str) and (
            reason_in in RATING_REASONS or reason_in in UPSTREAM_ERROR_REASONS
        ):
            reason = reason_in
        else:
            self.send_error(HTTPStatus.BAD_REQUEST, "reason is not a known value")
            return

        all_data = self._read_roof_ratings()
        if rating is None:
            all_data.pop(building_uuid, None)
            record: dict[str, Any] | None = None
        else:
            record = {"rating": rating, "updated_at": _ts()}
            if reason is not None:
                record["reason"] = reason
            all_data[building_uuid] = record
        self._write_roof_ratings(all_data)

        self._send_json(
            HTTPStatus.OK, {"ok": True, "uuid": building_uuid, "record": record}
        )

    # --- labeler endpoints ----------------------------------------------------

    def _handle_labeler_get(self, path: str, query: str) -> None:
        from reconcile_tiers.labeler import storage as lstore

        params = urllib.parse.parse_qs(query)
        # /labeler/runs                           → list
        # /labeler/runs/<run_id>                  → meta + count
        # /labeler/runs/<run_id>/case?index=N     → case + latest_label
        # /labeler/runs/<run_id>/labels           → all latest labels
        if path == "/labeler/runs":
            runs = [asdict(r) for r in lstore.list_runs()]
            self._send_json(HTTPStatus.OK, {"runs": runs})
            return

        parts = path[len("/labeler/runs/") :].split("/", 1)
        run_id = parts[0]
        if not _is_safe_run_id(run_id):
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid run_id")
            return
        rest = parts[1] if len(parts) > 1 else ""

        if rest == "":
            metas = {r.run_id: r for r in lstore.list_runs()}
            meta = metas.get(run_id)
            if meta is None:
                self.send_error(HTTPStatus.NOT_FOUND, "run not found")
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "meta": asdict(meta),
                    "case_count": lstore.case_count(run_id),
                    "labelled": len(lstore.latest_labels(run_id)),
                },
            )
            return

        if rest == "case":
            try:
                index = int(params.get("index", ["0"])[0])
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "index must be int")
                return
            case = lstore.get_case_at(run_id, index)
            if case is None:
                self.send_error(HTTPStatus.NOT_FOUND, "no case at index")
                return
            cid = case.get("case_id")
            label = lstore.latest_label(run_id, cid) if isinstance(cid, str) else None
            self._send_json(
                HTTPStatus.OK, {"index": index, "case": case, "label": label}
            )
            return

        if rest == "labels":
            self._send_json(HTTPStatus.OK, {"labels": lstore.latest_labels(run_id)})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "unknown labeler path")

    def _handle_labeler_post(self, path: str) -> None:
        from reconcile_tiers.labeler import storage as lstore
        from reconcile_tiers.labeler.schema import Label

        # POST /labeler/runs/<run_id>/labels
        if not path.startswith("/labeler/runs/"):
            self.send_error(HTTPStatus.NOT_FOUND, "unknown labeler path")
            return
        rest = path[len("/labeler/runs/") :]
        run_id, _, suffix = rest.partition("/")
        if not _is_safe_run_id(run_id) or suffix != "labels":
            self.send_error(
                HTTPStatus.BAD_REQUEST, "expected /labeler/runs/<run_id>/labels"
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid JSON body")
            return
        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "body must be an object")
            return

        case_id = payload.get("case_id")
        decision_type = payload.get("decision_type")
        kind = payload.get("label_kind", "select")
        selected = payload.get("selected_option_id")
        reasons = payload.get("reasons") or []
        labeler = payload.get("labeler") or os.environ.get("USER", "anon")
        if not isinstance(case_id, str) or not isinstance(decision_type, str):
            self.send_error(
                HTTPStatus.BAD_REQUEST, "case_id and decision_type required"
            )
            return
        if kind not in ("select", "skip", "unsure"):
            self.send_error(
                HTTPStatus.BAD_REQUEST, "label_kind must be select|skip|unsure"
            )
            return
        if kind == "select" and not isinstance(selected, str):
            self.send_error(
                HTTPStatus.BAD_REQUEST, "selected_option_id required for select"
            )
            return
        if not isinstance(reasons, list) or any(
            not isinstance(r, str) for r in reasons
        ):
            self.send_error(HTTPStatus.BAD_REQUEST, "reasons must be list of strings")
            return

        label = Label(
            case_id=case_id,
            decision_type=decision_type,
            selected_option_id=selected if kind == "select" else None,
            label_kind=kind,
            reasons=reasons,
            ts=time.time(),
            labeler=labeler,
        )
        lstore.append_label(run_id, label)
        self._send_json(HTTPStatus.OK, {"ok": True, "label": asdict(label)})


_SAFE_RUN_ID = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _is_safe_run_id(value: object) -> bool:
    if not isinstance(value, str) or not (1 <= len(value) <= 128):
        return False
    return all(c in _SAFE_RUN_ID for c in value)


# ---- entry point -------------------------------------------------------------


def serve(host: str = HOST, port: int = PORT) -> None:
    server = ThreadingHTTPServer((host, port), TierServerHandler)
    print(f"reconcile_tiers viewer on http://{host}:{port}{VIEWER_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
