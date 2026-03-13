#!/usr/bin/env python3
"""
AgentReview - Web UI

Flask app with real-time SSE streaming of the multi-agent review pipeline.
Supports file uploads (PDF + supplementary) and model selection.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    AssistantMessage,
    UserMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    TaskStartedMessage,
    TaskProgressMessage,
    TaskNotificationMessage,
)

from config import PipelineConfig
from agents.definitions import build_agent_definitions
from main import build_orchestrator_prompt


# ===================================================================
# APP SETUP
# ===================================================================

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "agentreview-dev")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

JOBS: dict[str, dict[str, Any]] = {}
JOB_QUEUES: dict[str, queue.Queue] = {}
JOB_STOP_EVENTS: dict[str, threading.Event] = {}

DEFAULT_CONFIG = PipelineConfig()
DEFAULT_CONFIG.ensure_dirs()

KNOWN_AGENTS = {
    "parser", "injection_detector", "novelty", "baselines",
    "code_reviewer", "soundness", "writing_quality", "reproducibility",
    "ethics_limitations", "synthesizer", "review_checker",
}

_AGENT_KEYWORDS: dict[str, list[str]] = {
    "parser":             ["parser", "parse", "extract", "pdf", "mineru", "markdown"],
    "injection_detector": ["injection", "inject", "hidden", "steganograph", "white text", "prompt injection"],
    "novelty":            ["novelty", "novel", "contribution", "prior work", "related work", "originality"],
    "baselines":          ["baseline", "missing baseline", "comparison", "state of the art", "sota", "leaderboard"],
    "code_reviewer":      ["code review", "code", "reproducib", "implementation"],
    "soundness":          ["soundness", "sound", "mathematical", "statistical", "rigor", "methodology", "experimental"],
    "synthesizer":        ["synthesiz", "synthesis", "final review", "openreview", "combine", "overall"],
    "review_checker":     ["review check", "checker", "verify", "validate review", "quality check", "factual accuracy"],
    "writing_quality":    ["writing", "grammar", "clarity", "presentation", "style", "figure", "table", "structure"],
    "reproducibility":    ["reproduc", "hyperparameter", "dataset", "training detail", "compute", "seed", "replicat"],
    "ethics_limitations": ["ethic", "limitation", "broader impact", "societal", "bias", "fairness", "privacy", "dual use"],
}


def _match_agent_from_description(text: str) -> str:
    if not text:
        return "subagent"
    text_lower = text.lower()
    best, best_score = "subagent", 0
    for agent_name, keywords in _AGENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best, best_score = agent_name, score
    return best


# ===================================================================
# EVENT HELPERS
# ===================================================================

def push_event(job_id: str, event_type: str, data: dict):
    q = JOB_QUEUES.get(job_id)
    if q:
        q.put({"type": event_type, **data})


_PATH_STRIP_PREFIXES = [
    str(Path(__file__).parent) + "/",
    str(Path.home()) + "/",
]


def _shorten_path(val: str) -> str:
    for prefix in _PATH_STRIP_PREFIXES:
        val = val.replace(prefix, "~/")
    val = val.replace("~/~/", "~/")
    return val


def _tool_input_summary(tool_input: dict) -> str:
    if not isinstance(tool_input, dict):
        return _shorten_path(str(tool_input)[:200])
    for key in ("file_path", "path", "command", "pattern", "prompt", "description", "query"):
        if key in tool_input:
            val = _shorten_path(str(tool_input[key]))
            return val[:120] + ("..." if len(val) > 120 else "")
    return _shorten_path(str(tool_input)[:200])


# ===================================================================
# PIPELINE RUNNER (background thread)
# ===================================================================

def _run_pipeline_bg(job_id: str, config: PipelineConfig, **kwargs):
    job = JOBS[job_id]
    workspace = Path(job["workspace"])

    # Set API key for this pipeline run (only if using api_key auth mode)
    api_key = kwargs.get("api_key")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    # If no API key, claude_agent_sdk uses existing Claude Code auth

    agent_models = kwargs.get("agent_models")
    model = kwargs.get("model")
    agents = build_agent_definitions(config, agent_models=agent_models)

    prompt = build_orchestrator_prompt(
        pdf_path=kwargs["pdf_path"],
        supplementary_paths=kwargs.get("supplementary_paths"),
        code_path=kwargs.get("code_path"),
        config=config,
        workspace=workspace,
        review_mode=kwargs.get("review_mode", "peer_review"),
    )

    (workspace / "orchestrator_prompt.txt").write_text(prompt)

    push_event(job_id, "stage", {"stage": "starting", "message": "Orchestrator agent starting..."})

    start_time = time.time()
    stop_event = JOB_STOP_EVENTS.get(job_id)

    async def _run():
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model=model,
                cwd=str(config.base_dir),
                allowed_tools=["Read", "Write", "Bash", "Glob", "Grep", "Agent", "WebSearch", "WebFetch"],
                agents=agents,
                permission_mode="acceptEdits",
                max_turns=200,
            ),
        ):
            if stop_event and stop_event.is_set():
                job["status"] = "stopped"
                push_event(job_id, "error", {"message": "Pipeline stopped by user."})
                break

            elapsed = time.time() - start_time

            if isinstance(message, ResultMessage):
                job["status"] = "completed"
                job["result"] = message.result
                cost = getattr(message, "total_cost_usd", None)
                if cost is not None:
                    job["total_cost_usd"] = cost
                result_usage = getattr(message, "usage", None)
                if isinstance(result_usage, dict):
                    input_tok = result_usage.get("input_tokens", 0) or 0
                    output_tok = result_usage.get("output_tokens", 0) or 0
                    result_total = input_tok + output_tok
                    if result_total > job.get("total_tokens", 0):
                        job["total_tokens"] = result_total
                push_event(job_id, "result", {
                    "message": message.result or "Pipeline complete.",
                    "elapsed": round(elapsed, 1),
                })

            elif isinstance(message, SystemMessage):
                sid = getattr(message, "session_id", None)
                if sid:
                    job["session_id"] = sid
                    push_event(job_id, "system", {"message": f"Session: {sid}"})

            elif isinstance(message, AssistantMessage):
                content = getattr(message, "content", [])
                for block in content:
                    if isinstance(block, ThinkingBlock):
                        text = getattr(block, "thinking", "")
                        if text:
                            push_event(job_id, "thinking", {
                                "message": text[:300] + ("..." if len(text) > 300 else ""),
                            })

                    elif isinstance(block, TextBlock):
                        text = getattr(block, "text", "")
                        if text:
                            push_event(job_id, "text", {"message": text})

                    elif isinstance(block, ToolResultBlock):
                        raw = getattr(block, "content", "")
                        if isinstance(raw, list):
                            parts = []
                            for item in raw:
                                if hasattr(item, "text"):
                                    parts.append(item.text)
                                elif isinstance(item, str):
                                    parts.append(item)
                            raw = "\n".join(parts)
                        if isinstance(raw, str) and raw:
                            truncated = raw[:8000] + ("\n... (truncated)" if len(raw) > 8000 else "")
                            push_event(job_id, "tool_result", {
                                "content": truncated,
                                "message": truncated[:200],
                            })

                    elif isinstance(block, ToolUseBlock):
                        name = getattr(block, "name", "?")
                        tool_input = getattr(block, "input", {})
                        summary = _tool_input_summary(tool_input)

                        if name == "Agent":
                            agent_type = tool_input.get("subagent_type") or tool_input.get("agent_type") or tool_input.get("type") or "subagent"
                            if agent_type == "subagent" or agent_type not in KNOWN_AGENTS:
                                agent_type = _match_agent_from_description(
                                    tool_input.get("description", "") + " " + tool_input.get("prompt", "")
                                )
                            desc = tool_input.get("description", "")
                            push_event(job_id, "agent", {
                                "agent": agent_type,
                                "description": desc,
                                "message": f"Invoking {agent_type} agent",
                            })
                        else:
                            detail = {}
                            if name == "Bash":
                                detail["command"] = _shorten_path(tool_input.get("command", ""))
                            elif name in ("Read", "Write", "Edit"):
                                detail["file_path"] = _shorten_path(tool_input.get("file_path", ""))
                                if name == "Write":
                                    content_str = str(tool_input.get("content", ""))
                                    detail["content_preview"] = content_str[:1000] + ("..." if len(content_str) > 1000 else "")
                            elif name in ("Grep", "Glob"):
                                detail["pattern"] = tool_input.get("pattern", "")
                                detail["path"] = _shorten_path(tool_input.get("path", ""))

                            push_event(job_id, "tool", {
                                "tool": name,
                                "summary": summary,
                                "detail": detail,
                                "message": f"{name}: {summary}",
                            })

            elif isinstance(message, UserMessage):
                content = getattr(message, "content", None)
                if isinstance(content, str) and content:
                    truncated = content[:8000] + ("\n... (truncated)" if len(content) > 8000 else "")
                    push_event(job_id, "tool_result", {
                        "content": truncated,
                        "message": truncated[:200],
                    })
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolResultBlock):
                            raw = getattr(block, "content", "")
                            if isinstance(raw, list):
                                parts = []
                                for item in raw:
                                    if isinstance(item, dict) and "text" in item:
                                        parts.append(item["text"])
                                    elif hasattr(item, "text"):
                                        parts.append(item.text)
                                    elif isinstance(item, str):
                                        parts.append(item)
                                raw = "\n".join(parts)
                            if isinstance(raw, str) and raw:
                                truncated = raw[:8000] + ("\n... (truncated)" if len(raw) > 8000 else "")
                                push_event(job_id, "tool_result", {
                                    "content": truncated,
                                    "message": truncated[:200],
                                })

            elif isinstance(message, TaskStartedMessage):
                agent_type = getattr(message, "task_type", None) or "subagent"
                desc = getattr(message, "description", "")
                if agent_type == "subagent" or agent_type not in KNOWN_AGENTS:
                    agent_type = _match_agent_from_description(desc)
                push_event(job_id, "task_started", {
                    "agent": agent_type,
                    "message": f"Subagent {agent_type} started",
                })

            elif isinstance(message, TaskProgressMessage):
                task_id = getattr(message, "task_id", None)
                usage = getattr(message, "usage", None)
                total_tok = 0
                if usage:
                    if isinstance(usage, dict):
                        total_tok = usage.get("total_tokens", 0) or 0
                    else:
                        total_tok = getattr(usage, "total_tokens", 0) or 0
                task_tokens = job.setdefault("task_tokens", {})
                if task_id and total_tok:
                    task_tokens[task_id] = total_tok
                job["total_tokens"] = sum(task_tokens.values())
                push_event(job_id, "task_progress", {
                    "total_tokens": job.get("total_tokens", 0),
                    "message": f"Progress: {job.get('total_tokens', 0)} tokens",
                })

            elif isinstance(message, TaskNotificationMessage):
                status = getattr(message, "status", None)
                status_str = getattr(status, "value", str(status)) if status else "done"
                task_id = getattr(message, "task_id", None)
                usage = getattr(message, "usage", None)
                if task_id and usage:
                    total_tok = 0
                    if isinstance(usage, dict):
                        total_tok = usage.get("total_tokens", 0) or 0
                    else:
                        total_tok = getattr(usage, "total_tokens", 0) or 0
                    if total_tok:
                        task_tokens = job.setdefault("task_tokens", {})
                        task_tokens[task_id] = total_tok
                        job["total_tokens"] = sum(task_tokens.values())
                push_event(job_id, "task_done", {
                    "status": status_str,
                    "message": f"Subagent finished: {status_str}",
                })

    try:
        asyncio.run(_run())
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        push_event(job_id, "error", {"message": str(e)})

    # Check for final review
    review_path = Path(job["workspace"]) / "final_review.md"
    if review_path.exists():
        job["review_md"] = review_path.read_text()
    review_json_path = Path(job["workspace"]) / "final_review.json"
    if review_json_path.exists():
        try:
            job["review_json"] = json.loads(review_json_path.read_text())
        except json.JSONDecodeError:
            pass

    elapsed = time.time() - start_time
    job["duration"] = round(elapsed, 1)

    push_event(job_id, "done", {
        "message": "Pipeline finished",
        "duration": round(elapsed, 1),
        "status": job.get("status", "completed"),
        "total_tokens": job.get("total_tokens", 0),
        "total_cost_usd": job.get("total_cost_usd"),
        "has_review": review_path.exists(),
    })


# ===================================================================
# ROUTES
# ===================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    # Handle file uploads
    paper_pdf = request.files.get("paper_pdf")
    if not paper_pdf or not paper_pdf.filename:
        return jsonify({"error": "Paper PDF is required"}), 400

    auth_mode = request.form.get("auth_mode", "api_key")
    api_key = request.form.get("api_key", "").strip()
    if auth_mode == "api_key" and not api_key:
        return jsonify({"error": "Anthropic API key is required"}), 400

    review_mode = request.form.get("review_mode", "peer_review")
    model_preset = request.form.get("model", "sonnet")
    enable_injection = request.form.get("enable_injection", "true") == "true"
    enable_novelty = request.form.get("enable_novelty", "true") == "true"
    enable_baselines = request.form.get("enable_baselines", "true") == "true"
    enable_soundness = request.form.get("enable_soundness", "true") == "true"
    enable_code_reviewer = request.form.get("enable_code_reviewer", "false") == "true"
    enable_writing_quality = request.form.get("enable_writing_quality", "true") == "true"
    enable_reproducibility = request.form.get("enable_reproducibility", "true") == "true"
    enable_ethics_limitations = request.form.get("enable_ethics_limitations", "true") == "true"

    # Create job
    job_id = uuid.uuid4().hex[:12]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + job_id[:6]

    config = PipelineConfig()
    config.enable_injection_detector = enable_injection
    config.enable_novelty = enable_novelty
    config.enable_baselines = enable_baselines
    config.enable_soundness = enable_soundness
    config.enable_code_reviewer = enable_code_reviewer
    config.enable_writing_quality = enable_writing_quality
    config.enable_reproducibility = enable_reproducibility
    config.enable_ethics_limitations = enable_ethics_limitations
    config.ensure_dirs()

    workspace = config.workspace_dir / run_id
    workspace.mkdir(parents=True, exist_ok=True)
    upload_dir = workspace / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save main paper
    paper_filename = secure_filename(paper_pdf.filename)
    paper_path = upload_dir / paper_filename
    paper_pdf.save(str(paper_path))

    # Save supplementary files
    supp_paths = []
    for f in request.files.getlist("supplementary"):
        if f and f.filename:
            fname = secure_filename(f.filename)
            fpath = upload_dir / fname
            f.save(str(fpath))
            supp_paths.append(str(fpath))

    # Check if any supplementary is code (zip)
    code_path = None
    for sp in supp_paths:
        if sp.endswith(".zip"):
            code_dir = workspace / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            import zipfile
            try:
                with zipfile.ZipFile(sp, "r") as zf:
                    zf.extractall(str(code_dir))
                code_path = str(code_dir)
                config.enable_code_reviewer = True
            except zipfile.BadZipFile:
                pass

    JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "run_id": run_id,
        "workspace": str(workspace),
        "paper_filename": paper_filename,
        "created": datetime.now().isoformat(),
        "total_tokens": 0,
    }
    JOB_QUEUES[job_id] = queue.Queue()
    JOB_STOP_EVENTS[job_id] = threading.Event()

    # Resolve model selection
    if model_preset in ("opus", "sonnet", "haiku"):
        orchestrator_model = model_preset
        agent_models = {"*": model_preset}
    else:
        orchestrator_model = "sonnet"
        agent_models = {"*": "sonnet"}

    t = threading.Thread(
        target=_run_pipeline_bg,
        args=(job_id, config),
        kwargs={
            "pdf_path": str(paper_path),
            "supplementary_paths": supp_paths if supp_paths else None,
            "code_path": code_path,
            "model": orchestrator_model,
            "agent_models": agent_models,
            "api_key": api_key,
            "review_mode": review_mode,
        },
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "run_id": run_id})


@app.route("/api/events/<job_id>")
def api_events(job_id):
    q = JOB_QUEUES.get(job_id)
    if not q:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/job/<job_id>")
def api_job(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/review/<job_id>")
def api_review(job_id):
    """Return the final review markdown for a completed job."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    review_md = job.get("review_md")
    review_json = job.get("review_json")
    if not review_md:
        review_path = Path(job["workspace"]) / "final_review.md"
        if review_path.exists():
            review_md = review_path.read_text()
    return jsonify({
        "review_md": review_md or "",
        "review_json": review_json or {},
    })


@app.route("/api/paper/<job_id>")
def api_paper(job_id):
    """Return the parsed paper markdown for display in UI."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    workspace = Path(job["workspace"])
    paper_md = ""
    metadata = {}
    paper_path = workspace / "parsed_paper.md"
    if paper_path.exists():
        paper_md = paper_path.read_text()
    meta_path = workspace / "paper_metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass
    return jsonify({
        "paper_md": paper_md,
        "metadata": metadata,
    })


@app.route("/api/jobs")
def api_jobs():
    return jsonify([
        {"id": j["id"], "status": j.get("status"), "paper": j.get("paper_filename"), "created": j.get("created")}
        for j in JOBS.values()
    ])


@app.route("/api/stop/<job_id>", methods=["POST"])
def api_stop(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "running":
        return jsonify({"error": "Job is not running"}), 400
    stop_event = JOB_STOP_EVENTS.get(job_id)
    if stop_event:
        stop_event.set()
    return jsonify({"ok": True})


# ===================================================================
# ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5008"))
    print(f"AgentReview - http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
