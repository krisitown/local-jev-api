"""Model-specific adapters for the non-Clef local decision checkpoints.

The adapters call publisher-authored inference code from either the pinned model
snapshot or runtime/vendor. No model downloads happen here; callers must pass a
complete, locally cached snapshot path. This module is loaded once per candidate
container, so each model's package names remain isolated.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

Request = dict[str, Any]
Response = dict[str, Any]
Predictor = Callable[[Request], Response]
_RUNTIME_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _RUNTIME_DIR / "vendor"


def _local_snapshot(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model snapshot directory does not exist: {path}")
    return path


def _add_import_root(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)
    importlib.invalidate_caches()


def _rows(request: Request) -> list[dict[str, Any]]:
    state = request["state"]
    questions = request["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    return [{"state": state, "question": q} for q in questions.values()]


def _backend_name(request: Request, snapshot: Path) -> str:
    return str(request.get("model") or snapshot.name)


def _load_pplx(snapshot: Path) -> Predictor:
    # The checkpoint itself contains the matching evaluated implementation.
    source = snapshot / "source" / "src"
    if not (source / "autojev" / "model.py").is_file():
        raise FileNotFoundError(f"PPLX snapshot is missing included inference source: {source}")
    _add_import_root(source)
    from autojev.model import DecisionModel, answer

    model = DecisionModel(snapshot, device="cuda")
    model.eval()

    def predict(request: Request) -> Response:
        questions = request["questions"]
        rows = _rows(request)
        # prepare() enforces the publisher's 8,192-token input limit and records
        # input_tokens. Forward/temperature/softmax/answer are the exact release path.
        batch = model.prepare(rows, max_length=8192)
        import torch

        with torch.inference_mode():
            probabilities = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
        answers = {
            key: answer(question, values[:count])
            for (key, question), values, count in zip(questions.items(), probabilities, batch.counts, strict=True)
        }
        return {
            "model": _backend_name(request, snapshot),
            "answers": answers,
            "usage": {"input_tokens": batch.input_tokens, "output_tokens": 0},
        }

    return predict


def _load_neohorse(snapshot: Path) -> Predictor:
    # This is the exact 1.0.0 runtime wheel shipped in the pinned HF snapshot,
    # unpacked under vendor/neohorse so startup needs no package installation.
    _add_import_root(_VENDOR_DIR / "neohorse")
    from neohorse_decision import DecisionEngine
    from neohorse_decision.systemone import format_response

    if not (snapshot / "backbone").is_dir() or not (snapshot / "model_manifest.json").is_file():
        raise FileNotFoundError(f"NeoHorse path must be the complete model bundle: {snapshot}")
    engine = DecisionEngine(str(snapshot), device="cuda")

    def predict(request: Request) -> Response:
        # DecisionEngine.predict validates model aliases, serializes GPU access,
        # and uses its exact strict local bundle loader. format_response applies
        # the publisher's confidence and output-token accounting for /v1/systemone.
        engine_request = {**request, "model": "NeoHorse-Jev-4B"}
        result = engine.predict(engine_request)
        response = format_response(result, engine.tokenizer)
        response["model"] = _backend_name(request, snapshot)
        return response

    return predict


def _load_jeff(snapshot: Path) -> Predictor:
    # Serving-only files copied from Jeff tag v1.2; no training/runtime extras.
    _add_import_root(_VENDOR_DIR / "jeff")
    from jeff.model import DecisionModel, answer

    model = DecisionModel(str(snapshot), device="cuda")
    model.eval()

    def predict(request: Request) -> Response:
        questions = request["questions"]
        rows = _rows(request)
        input_tokens = 0
        distributions: list[list[float]] = []
        import torch

        # This is Jeff v1.2's own server distributions path: exact trained
        # 8,192-token cap, microbatch size 8, and the checkpoint calibration
        # temperature applied once before converting probabilities to answers.
        with torch.inference_mode():
            for start in range(0, len(rows), 8):
                batch = model.prepare(rows[start : start + 8], max_length=8192)
                values = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
                distributions.extend(
                    row[:count]
                    for row, count in zip(values, batch.counts, strict=True)
                )
                input_tokens += batch.input_tokens
        answers = {
            key: answer(question, values)
            for (key, question), values in zip(questions.items(), distributions, strict=True)
        }
        return {
            "model": _backend_name(request, snapshot),
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0, "orders": 1},
        }

    return predict


def _load_laya(snapshot: Path) -> Predictor:
    # Laya 0.3.4's published load() accepts an existing local model directory.
    # Its Agent requires instructions; match laya-server's question-name fallback.
    _add_import_root(_VENDOR_DIR / "laya")
    import laya
    import shutil
    import tempfile
    # The publisher fixes tokenizer metadata in place. Keep the pinned snapshot
    # immutable by copying small metadata and symlinking only the weights.
    working = Path(tempfile.mkdtemp(prefix="laya-runtime-")) / "model"
    shutil.copytree(snapshot, working, ignore=shutil.ignore_patterns('*.safetensors', '.cache'))
    for weights in snapshot.rglob('*.safetensors'):
        target = working / weights.relative_to(snapshot)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(weights)
    agent = laya.load(str(working), device="cuda")

    def predict(request: Request) -> Response:
        questions: dict[str, Any] = {}
        for key, value in request["questions"].items():
            question = dict(value)
            if question.get("instructions") is None:
                question["instructions"] = key.replace("_", " ")
            questions[key] = question
        response = agent.system_one(request["state"], questions)
        response["model"] = _backend_name(request, snapshot)
        return response

    return predict


def load_backend(backend: str, localpath: str | Path) -> Predictor:
    """Load a local checkpoint and return ``predict(request) -> response``.

    Supported backend names are ``pplx``, ``neohorse``, ``jeff`` and ``laya``.
    The model directory must already exist locally; this function never invokes
    Hugging Face downloads or package installers.
    """
    snapshot = _local_snapshot(localpath)
    loaders = {
        "pplx": _load_pplx,
        "neohorse": _load_neohorse,
        "jeff": _load_jeff,
        "laya": _load_laya,
    }
    try:
        loader = loaders[backend]
    except KeyError as error:
        raise ValueError(f"Unsupported backend {backend!r}; choose one of {', '.join(loaders)}") from error
    return loader(snapshot)
