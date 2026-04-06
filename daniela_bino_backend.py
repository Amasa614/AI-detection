import os
import re
import threading
from dataclasses import dataclass
from typing import Iterator, List

import torch
import torch.nn.functional as F
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


def env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_OBSERVER_MODEL = os.getenv("BINO_OBSERVER_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
DEFAULT_PERFORMER_MODEL = os.getenv("BINO_PERFORMER_MODEL", "HuggingFaceTB/SmolLM2-135M")
DEFAULT_THRESHOLD = float(os.getenv("BINO_THRESHOLD", "0.901"))
DEFAULT_BORDERLINE_MARGIN = float(os.getenv("BINO_BORDERLINE_MARGIN", "0.03"))
DEFAULT_MAX_LENGTH = int(os.getenv("BINO_MAX_LENGTH", "128"))
DEFAULT_BATCH_SIZE = int(os.getenv("BINO_BATCH_SIZE", "1"))
DEFAULT_WINDOW_SIZE = int(os.getenv("BINO_WINDOW_SIZE", "6"))
DEFAULT_REWRITE_CANDIDATES = int(os.getenv("BINO_REWRITE_CANDIDATES", "2"))
DEFAULT_REWRITE_MAX_NEW_TOKENS = int(os.getenv("BINO_REWRITE_MAX_NEW_TOKENS", "60"))
# Separate budget for generation prompts (prompt + sentence must fit here)
DEFAULT_GENERATION_MAX_LENGTH = int(os.getenv("BINO_GENERATION_MAX_LENGTH", "192"))
SENTENCE_MIN_WORDS = int(os.getenv("BINO_MIN_SENTENCE_WORDS", "5"))
ENABLE_CPU_QUANTIZATION = env_flag("BINO_ENABLE_CPU_QUANTIZATION", "true")
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


def maybe_quantize_model(model: nn.Module, device: str) -> tuple[nn.Module, bool]:
    if device != "cpu" or not ENABLE_CPU_QUANTIZATION:
        return model, False
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model, True


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
    quantized: bool


class ScoreRequest(BaseModel):
    text: str = Field(min_length=1)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=8)
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=16, le=1024)


class RewriteRequest(BaseModel):
    sentences: List[str] = Field(min_length=1, max_length=5)
    candidate_count: int = Field(default=DEFAULT_REWRITE_CANDIDATES, ge=1, le=5)
    max_new_tokens: int = Field(default=DEFAULT_REWRITE_MAX_NEW_TOKENS, ge=16, le=160)
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=16, le=1024)


class AnalyseRequest(BaseModel):
    sentences: List[str] = Field(min_length=1, max_length=8)
    bino_scores: List[float] = Field(default=[], max_length=8)
    max_new_tokens: int = Field(default=60, ge=10, le=120)
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

            observer_quantized = False
            performer_quantized = False
            if device == "cpu":
                observer_model, observer_quantized = maybe_quantize_model(observer_model, device)
                performer_model, performer_quantized = maybe_quantize_model(performer_model, device)
            else:
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
                dtype="int8-dynamic" if observer_quantized or performer_quantized else str(dtype).replace("torch.", ""),
                quantized=observer_quantized or performer_quantized,
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


def worst_window_score(scores: List[float], window_size: int = DEFAULT_WINDOW_SIZE) -> float:
    if not scores:
        return float("nan")
    actual_window = max(1, min(window_size, len(scores)))
    windows = [
        sum(scores[index:index + actual_window]) / actual_window
        for index in range(0, len(scores) - actual_window + 1)
    ]
    return min(windows)


def build_rewrite_chat_prompt(tokenizer, sentence: str) -> str:
    """
    Minimal prompt for a small instruct model (135M).
    No system message - they consume too many tokens and the model echoes them.
    Single user turn, very short instruction so the full sentence fits in context.
    """
    messages = [
        {"role": "user", "content": f'Paraphrase for academic writing:\n"{sentence.strip()}"'},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f'Paraphrase for academic writing:\n"{sentence.strip()}"\n\nParaphrase:'


def is_likely_citation(sentence: str) -> bool:
    """Return True for reference-list entries that should not be rewritten."""
    s = sentence.strip()
    # Matches "Author, F., Author2, J., ..." style reference entries
    if re.match(r"^[A-Z][a-z]+,\s+[A-Z]\.", s):
        return True
    # More than 4 comma-separated author-like tokens
    if len(s.split(",")) > 5:
        return True
    # Contains a DOI or URL
    if re.search(r"https?://|doi\.org|doi:", s, re.IGNORECASE):
        return True
    return False


# Words that appear in the prompt itself — if the model echoes them, the output is junk.
_PROMPT_ECHO_SIGNALS = [
    "paraphrase", "academic writing", "academic style", "academic tone",
    "rewrite", "rephrase", "more natural", "naturally human",
    "sound natural", "same meaning", "original meaning",
    "conversational and natural", "maintain an academic",
    "no explanation", "no prefix",
]

_STOP_WORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "it", "that", "this", "be",
    "have", "has", "had", "not", "no", "by", "as", "from", "its",
])


def _content_words(text: str) -> set:
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower())) - _STOP_WORDS


def is_prompt_echo(candidate: str) -> bool:
    lower = candidate.lower()
    return any(signal in lower for signal in _PROMPT_ECHO_SIGNALS)


def has_content_overlap(original: str, candidate: str, min_ratio: float = 0.15) -> bool:
    """Ensure the rewrite shares at least some content words with the original."""
    orig = _content_words(original)
    cand = _content_words(candidate)
    if not orig:
        return True
    return len(orig & cand) / len(orig) >= min_ratio


def clean_rewrite_text(text: str) -> str:
    candidate = text.strip().strip('"').strip("'")
    # Take only the first non-empty line
    for line in candidate.splitlines():
        line = line.strip()
        if line:
            candidate = line
            break
    # Strip residual instruction prefixes
    candidate = re.sub(
        r"^(Paraphrase:|Rewrite:|Rewritten( sentence)?:|Here( is|'s)( the)? (rewrite|paraphrase):)\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = candidate.strip().strip('"').strip("'")
    if candidate and candidate[-1] not in ".!?":
        candidate += "."
    return candidate


def generate_rewrite_candidates(
    bundle: LoadedModels,
    sentence: str,
    candidate_count: int,
    max_new_tokens: int,
) -> List[str]:
    tokenizer = bundle.tokenizer
    prompt = build_rewrite_chat_prompt(tokenizer, sentence)

    # Use the generation budget so the full sentence fits without blowing RAM.
    prompt_max = DEFAULT_GENERATION_MAX_LENGTH
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_max,
        return_token_type_ids=False,
    )
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[-1]

    candidates: List[str] = []
    original_lower = sentence.strip().lower()
    with torch.inference_mode():
        for _ in range(candidate_count):
            generated = bundle.observer_model.generate(
                **encoded,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.25,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=4,
            )
            text = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
            cleaned = clean_rewrite_text(text)
            if (
                cleaned
                and len(cleaned) > 10
                and cleaned.lower() != original_lower
                and not is_prompt_echo(cleaned)
                and has_content_overlap(sentence, cleaned)
            ):
                candidates.append(cleaned)

    return list(dict.fromkeys(candidates))


def build_rewrite_suggestions(bundle: LoadedModels, request: RewriteRequest) -> List[dict]:
    suggestions: List[dict] = []
    for sentence in request.sentences:
        original_sentence = sentence.strip()
        if not original_sentence:
            continue
        if is_likely_citation(original_sentence):
            suggestions.append({
                "original_sentence": original_sentence,
                "original_bino": None,
                "rewritten_sentence": None,
                "rewritten_bino": None,
                "improved": False,
                "passes_threshold": False,
                "candidate_count": 0,
                "skipped": True,
                "skip_reason": "citation or reference entry",
            })
            continue

        original_result = score_batch(bundle, [original_sentence], request.max_length)[0]
        candidates = generate_rewrite_candidates(
            bundle,
            original_sentence,
            request.candidate_count,
            request.max_new_tokens,
        )

        scored_candidates = []
        for candidate in candidates:
            if sentence_word_count(candidate) < SENTENCE_MIN_WORDS:
                continue
            scored = score_batch(bundle, [candidate], request.max_length)[0]
            scored_candidates.append(
                {
                    "rewrite": candidate,
                    "bino": scored["bino"],
                    "ppl": scored["ppl"],
                    "xppl": scored["xppl"],
                }
            )

        scored_candidates.sort(key=lambda item: item["bino"], reverse=True)
        best = scored_candidates[0] if scored_candidates else None
        suggestions.append(
            {
                "original_sentence": original_sentence,
                "original_bino": original_result["bino"],
                "rewritten_sentence": best["rewrite"] if best else None,
                "rewritten_bino": best["bino"] if best else None,
                "improved": bool(best and best["bino"] > original_result["bino"]),
                "passes_threshold": bool(best and best["bino"] >= DEFAULT_THRESHOLD),
                "candidate_count": len(scored_candidates),
            }
        )

    return suggestions


def verdict_for(score: float, sentence_scores: List[float]) -> str:
    if not sentence_scores:
        return "ai"

    ai_sentence_count = sum(item < DEFAULT_THRESHOLD for item in sentence_scores)
    low_ratio = ai_sentence_count / len(sentence_scores)
    worst_window = worst_window_score(sentence_scores)
    borderline_ceiling = DEFAULT_THRESHOLD + DEFAULT_BORDERLINE_MARGIN

    if score < DEFAULT_THRESHOLD or low_ratio >= 0.15 or worst_window < DEFAULT_THRESHOLD:
        return "ai"
    if score < borderline_ceiling or ai_sentence_count > 0:
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
        "borderline_ceiling": DEFAULT_THRESHOLD + DEFAULT_BORDERLINE_MARGIN,
        "batch_size": DEFAULT_BATCH_SIZE,
        "max_length": DEFAULT_MAX_LENGTH,
        "cpu_quantization": ENABLE_CPU_QUANTIZATION,
    }
    if loaded:
        payload["device"] = store._models.device
        payload["dtype"] = store._models.dtype
        payload["quantized"] = store._models.quantized
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
    sentence_bino_scores = [item["bino"] for item in valid_scores]
    ai_sentence_count = sum(item < DEFAULT_THRESHOLD for item in sentence_bino_scores)
    borderline_sentence_count = sum(
        DEFAULT_THRESHOLD <= item < (DEFAULT_THRESHOLD + DEFAULT_BORDERLINE_MARGIN)
        for item in sentence_bino_scores
    )
    lowest_sentence_score = min(sentence_bino_scores)
    lowest_window_score = worst_window_score(sentence_bino_scores)

    return {
        "observer_model": bundle.observer_name,
        "performer_model": bundle.performer_name,
        "device": bundle.device,
        "dtype": bundle.dtype,
        "quantized": bundle.quantized,
        "threshold": DEFAULT_THRESHOLD,
        "borderline_ceiling": DEFAULT_THRESHOLD + DEFAULT_BORDERLINE_MARGIN,
        "sentence_count": len(sentences),
        "scored_sentence_count": len(valid_scores),
        "ai_sentence_count": ai_sentence_count,
        "borderline_sentence_count": borderline_sentence_count,
        "lowest_sentence_score": lowest_sentence_score,
        "lowest_window_score": lowest_window_score,
        "window_size": min(DEFAULT_WINDOW_SIZE, len(sentence_bino_scores)),
        "document_score": document_score,
        "document_verdict": verdict_for(document_score, sentence_bino_scores),
        "sentence_scores": valid_scores,
        "notes": [
            "This backend uses a low-memory SmolLM2 observer/performer pair to keep the local Binoculars path lighter.",
            "On CPU it can apply dynamic int8 quantization and conservative defaults (batch size 1, max length 128).",
            "The Binoculars paper reports a global threshold around 0.901, but this smaller model pair is only an approximation and may need re-calibration.",
            "The first request may take a while because the models need to download and load.",
        ],
    }


_PATTERN_LABELS = {
    "formulaic": ["formulaic", "formula", "template", "structured", "boilerplate", "generic opener", "generic structure"],
    "no-personal-voice": ["personal voice", "no voice", "impersonal", "third person", "lacks first", "no first person", "missing voice"],
    "ungrounded": ["ungrounded", "generic claim", "vague", "no evidence", "no example", "no specific", "abstract", "lacks specificity", "unsupported"],
    "uniform-rhythm": ["uniform", "rhythm", "same length", "similar length", "monotone", "flat", "repetitive structure", "uniform sentence"],
    "over-hedged": ["over-hedge", "excessive hedge", "overly cautious", "weak claim", "too hedged", "too much hedging"],
    "missing-contrast": ["no contrast", "missing contrast", "no counterargument", "one-sided", "no nuance", "lacks nuance"],
    "citation-style": ["citation", "reference list", "bibliography", "reference entry"],
}


def detect_pattern_label(text: str) -> str:
    """Map model output to the closest predefined pattern label."""
    lower = text.lower()
    for label, keywords in _PATTERN_LABELS.items():
        if any(kw in lower for kw in keywords):
            return label
    return "ai-pattern"


_ANALYSE_SYSTEM = (
    "You are an expert academic writing analyst grading student essays, act like turnitin would. "
    "When shown a sentence that scored low on an AI-detection metric, "
    "identify in one concise sentence exactly what writing pattern makes it sound AI-generated. "
    "Name a specific pattern such as: formulaic opener, no personal stance, uniform rhythm, "
    "ungrounded claim, missing contrast, over-hedging, or citation-style phrasing. "
    "Do not repeat the sentence. Do not explain what to do. Just name and briefly describe the pattern."
)


def analyse_sentence(
    bundle: LoadedModels,
    sentence: str,
    bino_score: float,
    max_new_tokens: int,
) -> dict:
    tokenizer = bundle.tokenizer
    score_note = f"(bino score: {bino_score:.3f})" if isinstance(bino_score, float) else ""
    user_content = (
        f"Academic sentence flagged as likely AI-generated {score_note}:\n\n"
        f"\"{sentence.strip()}\"\n\n"
        "What specific AI writing pattern does this sentence show?"
    )
    messages = [
        {"role": "system", "content": _ANALYSE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"{_ANALYSE_SYSTEM}\n\n{user_content}\n\nPattern:"

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=bundle_max_length(bundle),
        return_token_type_ids=False,
    )
    encoded = {k: v.to(bundle.device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[-1]

    with torch.inference_mode():
        generated = bundle.observer_model.generate(
            **encoded,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=4,
        )

    raw = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()
    # Take only the first sentence of the explanation
    explanation = raw.splitlines()[0].strip()
    explanation = re.split(r"(?<=[.!?])\s", explanation)[0].strip()
    if explanation and explanation[-1] not in ".!?":
        explanation += "."

    return {
        "sentence": sentence.strip(),
        "bino_score": bino_score,
        "pattern_label": detect_pattern_label(explanation),
        "explanation": explanation if explanation else "No clear pattern identified.",
    }


def bundle_max_length(bundle: LoadedModels) -> int:
    return DEFAULT_GENERATION_MAX_LENGTH


@app.post("/analyse")
def analyse_sentences(request: AnalyseRequest) -> dict:
    sentences = [s.strip() for s in request.sentences if s.strip()]
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences provided.")

    # Align bino scores with sentences (pad with NaN if missing)
    scores = list(request.bino_scores) + [float("nan")] * len(sentences)
    scores = scores[: len(sentences)]

    try:
        bundle = store.get()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}") from exc

    results = []
    try:
        for sentence, bino in zip(sentences, scores):
            if is_likely_citation(sentence):
                continue
            result = analyse_sentence(
                bundle, sentence, bino, request.max_new_tokens
            )
            results.append(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return {
        "observer_model": bundle.observer_name,
        "device": bundle.device,
        "dtype": bundle.dtype,
        "threshold": DEFAULT_THRESHOLD,
        "analyses": results,
    }


@app.post("/rewrite-suggestions")
def rewrite_suggestions(request: RewriteRequest) -> dict:
    sentences = [sentence.strip() for sentence in request.sentences if sentence.strip()]
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences were provided.")

    try:
        bundle = store.get()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}") from exc

    try:
        rewrites = build_rewrite_suggestions(bundle, request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rewrite generation failed: {exc}") from exc

    return {
        "observer_model": bundle.observer_name,
        "performer_model": bundle.performer_name,
        "device": bundle.device,
        "dtype": bundle.dtype,
        "quantized": bundle.quantized,
        "threshold": DEFAULT_THRESHOLD,
        "borderline_ceiling": DEFAULT_THRESHOLD + DEFAULT_BORDERLINE_MARGIN,
        "rewrite_suggestions": rewrites,
    }
