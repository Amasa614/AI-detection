import os
import re
import threading
from dataclasses import dataclass
from typing import Iterator, List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_OBSERVER_MODEL = os.getenv("BINO_OBSERVER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_PERFORMER_MODEL = os.getenv("BINO_PERFORMER_MODEL", "Qwen/Qwen2.5-0.5B")
DEFAULT_THRESHOLD = float(os.getenv("BINO_THRESHOLD", "0.9"))
DEFAULT_MAX_LENGTH = int(os.getenv("BINO_MAX_LENGTH", "256"))
DEFAULT_BATCH_SIZE = int(os.getenv("BINO_BATCH_SIZE", "2"))
SENTENCE_MIN_WORDS = int(os.getenv("BINO_MIN_SENTENCE_WORDS", "5"))
CORS_ORIGINS = [origin.strip() for origin in os.getenv("BINO_CORS_ORIGINS", "*").split(",") if origin.strip()]
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def sentence_word_count(text: str) -> int:
    return len([part for part in re.split(r"\s+", text.strip()) if part])


def get_sentences(text: str, min_words: int = SENTENCE_MIN_WORDS) -> List[str]:
    raw = [part.strip() for part in SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]
    return [sentence for sentence in raw if sentence_word_count(sentence) >= min_words]


def batched(items: List[str], size: int) -> Iterator[List[str]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


@dataclass
class LoadedModels:
    tokenizer: object
    observer_model: object
    performer_model: object
    observer_name: str
    performer_name: str
    device: str
    dtype: str


class ScoreRequest(BaseModel):
    text: str = Field(min_length=1)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=8)
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=16, le=1024)


class ModelStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: LoadedModels | None = None

    def get(self) -> LoadedModels:
        if self._models is not None:
            return self._models

        with self._lock:
            if self._models is not None:
                return self._models

            device = pick_device()
            dtype = pick_dtype(device)

            observer_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_OBSERVER_MODEL, use_fast=True)
            performer_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PERFORMER_MODEL, use_fast=True)

            if observer_tokenizer.get_vocab() != performer_tokenizer.get_vocab():
                raise RuntimeError(
                    "Observer and performer tokenizers do not match. Choose a model pair with the same tokenizer."
                )

            if observer_tokenizer.pad_token is None:
                observer_tokenizer.pad_token = observer_tokenizer.eos_token

            observer_model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_OBSERVER_MODEL,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            performer_model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_PERFORMER_MODEL,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )

            observer_model.to(device)
            performer_model.to(device)
            observer_model.eval()
            performer_model.eval()

            self._models = LoadedModels(
                tokenizer=observer_tokenizer,
                observer_model=observer_model,
                performer_model=performer_model,
                observer_name=DEFAULT_OBSERVER_MODEL,
                performer_name=DEFAULT_PERFORMER_MODEL,
                device=device,
                dtype=str(dtype).replace("torch.", ""),
            )
            return self._models


store = ModelStore()
app = FastAPI(title="Daniela Binoculars Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def score_batch(bundle: LoadedModels, sentences: List[str], max_length: int) -> List[dict]:
    tokenizer = bundle.tokenizer
    encoded = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
    )
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}

    with torch.inference_mode():
        observer_logits = bundle.observer_model(**encoded).logits[..., :-1, :]
        performer_logits = bundle.performer_model(**encoded).logits[..., :-1, :]

        shifted_ids = encoded["input_ids"][..., 1:]
        shifted_mask = encoded["attention_mask"][..., 1:].to(observer_logits.dtype)

        observer_log_probs = F.log_softmax(observer_logits, dim=-1)
        performer_log_probs = F.log_softmax(performer_logits, dim=-1)
        observer_probs = observer_log_probs.exp()

        ppl_tokens = -observer_log_probs.gather(-1, shifted_ids.unsqueeze(-1)).squeeze(-1)
        xppl_tokens = -(observer_probs * performer_log_probs).sum(dim=-1)

        denom = shifted_mask.sum(dim=-1).clamp_min(1)
        ppl = (ppl_tokens * shifted_mask).sum(dim=-1) / denom
        xppl = (xppl_tokens * shifted_mask).sum(dim=-1) / denom
        bino = ppl / xppl

    results = []
    for idx, sentence in enumerate(sentences):
        token_count = int(denom[idx].item())
        ppl_value = float(ppl[idx].detach().cpu().item())
        xppl_value = float(xppl[idx].detach().cpu().item())
        bino_value = float(bino[idx].detach().cpu().item())
        results.append(
            {
                "sentence": sentence,
                "token_count": token_count,
                "ppl": ppl_value,
                "xppl": xppl_value,
                "bino": bino_value,
            }
        )

    if bundle.device == "cuda":
        torch.cuda.empty_cache()

    return results


def verdict_for(score: float) -> str:
    if score < DEFAULT_THRESHOLD:
        return "ai"
    if score < 1.1:
        return "borderline"
    return "human"


@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "message": "Daniela Binoculars backend is running.",
        "health_url": "/health",
        "docs_url": "/docs",
        "score_url": "/score",
    }


@app.get("/health")
def health() -> dict:
    loaded = store._models is not None
    payload = {
        "status": "ok",
        "loaded": loaded,
        "cors_origins": CORS_ORIGINS or ["*"],
        "observer_model": DEFAULT_OBSERVER_MODEL,
        "performer_model": DEFAULT_PERFORMER_MODEL,
        "threshold": DEFAULT_THRESHOLD,
    }
    if loaded:
        payload["device"] = store._models.device
        payload["dtype"] = store._models.dtype
    return payload


@app.post("/score")
def score(request: ScoreRequest) -> dict:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    sentences = get_sentences(text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No scorable sentences found.")

    try:
        bundle = store.get()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}") from exc

    all_results: List[dict] = []
    try:
        for batch in batched(sentences, request.batch_size):
            all_results.extend(score_batch(bundle, batch, request.max_length))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc

    valid_scores = [
        item for item in all_results
        if torch.isfinite(torch.tensor(item["bino"])) and torch.isfinite(torch.tensor(item["xppl"]))
    ]
    if not valid_scores:
        raise HTTPException(status_code=500, detail="No valid sentence scores were produced.")

    document_score = sum(item["bino"] for item in valid_scores) / len(valid_scores)

    return {
        "observer_model": bundle.observer_name,
        "performer_model": bundle.performer_name,
        "device": bundle.device,
        "dtype": bundle.dtype,
        "threshold": DEFAULT_THRESHOLD,
        "sentence_count": len(sentences),
        "scored_sentence_count": len(valid_scores),
        "document_score": document_score,
        "document_verdict": verdict_for(document_score),
        "sentence_scores": valid_scores,
        "notes": [
            "This backend uses full next-token distributions from two open models with the same tokenizer.",
            "Thresholds are model-pair dependent; 0.9 is a starting reference, not a calibrated guarantee.",
            "The first request may take a while because the models need to download and load.",
        ],
    }
