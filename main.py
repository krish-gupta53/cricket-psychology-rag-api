import json
import os
import hashlib
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional
from operator import add

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator
from pydantic import ConfigDict
from openai import OpenAI
from pinecone import Pinecone

from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# OpenAI pricing table  (USD per 1M tokens, as of April 2026)
# Update these if prices change.
# ---------------------------------------------------------------------------

_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4.1":          {"input": 2.00,  "output": 8.00},
    "gpt-4.1-mini":     {"input": 0.40,  "output": 1.60},
    "gpt-4.1-nano":     {"input": 0.10,  "output": 0.40},
    "gpt-4o":           {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    # embedding models (only input tokens apply)
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


# ---------------------------------------------------------------------------
# Token tracker  (one instance per request, passed via PipelineState)
# ---------------------------------------------------------------------------

@dataclass
class NodeUsage:
    node: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class TokenTracker:
    entries: List[NodeUsage] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def record(self, node: str, model: str, usage: Any) -> None:
        """Record usage from an OpenAI completion response.usage object."""
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
        pricing = _MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (
            prompt_tokens * pricing["input"] / 1_000_000
            + completion_tokens * pricing["output"] / 1_000_000
        )
        self.entries.append(NodeUsage(
            node=node,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
        ))

    def record_embedding(self, node: str, model: str, token_count: int) -> None:
        """Record usage from an OpenAI embedding response."""
        pricing = _MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = token_count * pricing["input"] / 1_000_000
        self.entries.append(NodeUsage(
            node=node,
            model=model,
            prompt_tokens=token_count,
            completion_tokens=0,
            total_tokens=token_count,
            cost_usd=cost,
        ))

    def totals(self) -> Dict[str, Any]:
        total_prompt = sum(e.prompt_tokens for e in self.entries)
        total_completion = sum(e.completion_tokens for e in self.entries)
        total_tokens = sum(e.total_tokens for e in self.entries)
        total_cost = sum(e.cost_usd for e in self.entries)
        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Settings  (per-node model config)
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_host: str
    pinecone_namespace: str = "content-automate"
    embedding_model: str = "text-embedding-3-small"

    # Per-node model overrides
    brief_model: str = "gpt-4.1-mini"
    query_model: str = "gpt-4.1-mini"
    angle_model: str = "gpt-4.1"
    final_model: str = "gpt-4.1"
    summary_model: str = "gpt-4.1-mini"

    top_k_per_query: int = 5
    final_chunk_count: int = 9

    @classmethod
    def from_env(cls) -> "Settings":
        required = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
            "pinecone_index_host": os.getenv("PINECONE_INDEX_HOST"),
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        return cls(
            openai_api_key=required["openai_api_key"],
            pinecone_api_key=required["pinecone_api_key"],
            pinecone_index_host=required["pinecone_index_host"],
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "content-automate"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            brief_model=os.getenv("BRIEF_MODEL", "gpt-4.1-mini"),
            query_model=os.getenv("QUERY_MODEL", "gpt-4.1-mini"),
            angle_model=os.getenv("ANGLE_MODEL", "gpt-4.1"),
            final_model=os.getenv("FINAL_MODEL", "gpt-4.1"),
            summary_model=os.getenv("SUMMARY_MODEL", "gpt-4.1-mini"),
            top_k_per_query=int(os.getenv("TOP_K_PER_QUERY", "5")),
            final_chunk_count=int(os.getenv("FINAL_CHUNK_COUNT", "9")),
        )


# ---------------------------------------------------------------------------
# API-Key auth
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str = Security(_api_key_header)) -> bool:
    expected = os.getenv("APP_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="Server auth not configured")
    if not api_key or not secrets.compare_digest(api_key.strip(), expected.strip()):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing X-API-Key header")
    return True


# ---------------------------------------------------------------------------
# FastAPI request/response models
# ---------------------------------------------------------------------------

class GenerateContentRequest(BaseModel):
    problem: str = Field(..., min_length=5)
    category: str = Field(..., min_length=2)
    sub_category: str = Field(..., min_length=2)
    cricket_role: str = Field(..., min_length=2)
    player_level: str = Field(..., min_length=2)
    psychology_angle: str = Field(..., min_length=2)
    thread_id: str = Field(default="default", description="Session thread ID for memory continuity")
    debug: bool = False


class QueryItem(BaseModel):
    query_type: Literal["symptom", "mental_skill", "deep_psychology"]
    query: str


class BriefResponse(BaseModel):
    core_problem: str
    player_situation: str
    emotional_truth: str
    hidden_cause: str
    wrong_advice_players_usually_get: str
    better_reframe: str
    desired_shift: str
    ideal_content_direction: str
    search_intent: str
    must_include: List[str]
    must_avoid: List[str]


class QueryGenerationResponse(BaseModel):
    queries: List[QueryItem]


# ---------------------------------------------------------------------------
# Angle models — resilient to LLM field-name drift
# ---------------------------------------------------------------------------

_ANGLE_OPTION_KEY_MAP: Dict[str, str] = {
    "why_it_works":        "why_it_could_work",
    "why_this_could_work": "why_it_could_work",
    "why_it_might_work":   "why_it_could_work",
    "reason":              "why_it_could_work",
    "rationale":           "why_it_could_work",
    "why":                 "why_it_could_work",
    "idea":                "one_line_idea",
    "one_line":            "one_line_idea",
    "summary":             "one_line_idea",
    "description":         "one_line_idea",
}

_CHOSEN_ANGLE_KEY_MAP: Dict[str, str] = {
    "why_it_is_best":       "why_this_is_best",
    "why_best":             "why_this_is_best",
    "rationale":            "why_this_is_best",
    "reason":               "why_this_is_best",
    "why":                  "why_this_is_best",
    "why_this_works_best":  "why_this_is_best",
    "promise":              "practical_promise",
    "practical_outcome":    "practical_promise",
    "outcome":              "practical_promise",
    "hook":                 "emotional_hook",
    "emotion":              "emotional_hook",
}


def _remap_keys(d: Dict[str, Any], key_map: Dict[str, str]) -> Dict[str, Any]:
    result = dict(d)
    for variant, canonical in key_map.items():
        if variant in result and canonical not in result:
            result[canonical] = result.pop(variant)
    return result


class AngleOption(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    angle_name: str
    one_line_idea: str
    why_it_could_work: str
    risk: str

    @model_validator(mode="before")
    @classmethod
    def normalise_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return _remap_keys(data, _ANGLE_OPTION_KEY_MAP)
        return data


class ChosenAngle(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    angle_name: str
    core_message: str
    emotional_hook: str
    practical_promise: str
    why_this_is_best: str

    @model_validator(mode="before")
    @classmethod
    def normalise_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return _remap_keys(data, _CHOSEN_ANGLE_KEY_MAP)
        return data


class AngleSelectionResponse(BaseModel):
    angle_options: List[AngleOption]
    chosen_angle: ChosenAngle


# ---------------------------------------------------------------------------
# Final content models
# ---------------------------------------------------------------------------

class CarouselSlide(BaseModel):
    slide_number: int
    slide_text: str


class MentalDrill(BaseModel):
    name: str
    when_to_use: str
    steps: List[str]


class SourceGrounding(BaseModel):
    retrieved_chunk_ids_used: List[str]
    key_ideas_used: List[str]


class FinalContentResponse(BaseModel):
    topic: str
    content_title: str
    core_insight: str
    hook_options: List[str]
    primary_hook: str
    reel_script: Dict[str, str]
    short_reel_script: str
    caption: str
    carousel_slides: List[CarouselSlide]
    mental_drill: MentalDrill
    one_line_takeaway: str
    cta: str
    hashtags: List[str]
    source_grounding: SourceGrounding


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    payload: Dict[str, Any]
    messages: Annotated[List[Any], add_messages]
    brief_summary: str
    brief: Optional[Dict[str, Any]]
    queries: Optional[List[Dict[str, Any]]]
    raw_chunks: Annotated[List[Dict[str, Any]], add]
    selected_chunks: Optional[List[Dict[str, Any]]]
    angle_selection: Optional[Dict[str, Any]]
    final_content: Optional[Dict[str, Any]]
    tracker: Optional[Any]   # TokenTracker instance — not serialised by LangGraph, passed by ref


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BRIEF_PROMPT = """You are a cricket psychology content strategist.

Your task is to enrich the selected cricket problem into a clear content brief.

{summary_context}

Input:

Problem:
{problem}

Category:
{category}

Sub-category:
{sub_category}

Cricket role:
{cricket_role}

Player level:
{player_level}

Psychology angle:
{psychology_angle}

Create a useful brief for content generation.

Return only valid JSON with this schema:

{{
  "core_problem": "",
  "player_situation": "",
  "emotional_truth": "",
  "hidden_cause": "",
  "wrong_advice_players_usually_get": "",
  "better_reframe": "",
  "desired_shift": "",
  "ideal_content_direction": "",
  "search_intent": "",
  "must_include": [],
  "must_avoid": []
}}

Rules:
- Be specific to cricket.
- Do not use generic motivational advice.
- Explain what the player is actually experiencing internally.
- Keep the language simple and practical.
- The output will be used to generate Pinecone search queries and final content.
- Return JSON only."""


QUERY_PROMPT = """You are a retrieval query expert for a cricket sports psychology RAG system.

Your task is to generate 3 different semantic search queries for Pinecone.

The goal is to retrieve the best knowledge chunks for creating cricket psychology content.

Input:

Original problem:
{problem}

Cricket role:
{cricket_role}

Player level:
{player_level}

Psychology angle:
{psychology_angle}

Category:
{category}

Content brief:
{content_brief}

Generate exactly 3 search queries.

Each query must search from a different angle:

1. Symptom query:
Search the exact player problem in natural cricket language.

2. Mental skill query:
Search for practical mental tools, routines, exercises, or interventions.

3. Deep psychology query:
Search for the underlying psychological pattern, such as fear, self-doubt, pressure, identity, confidence, focus, resilience, self-talk, or emotional control.

Return only valid JSON with this schema:

{{
  "queries": [
    {{"query_type": "symptom", "query": ""}},
    {{"query_type": "mental_skill", "query": ""}},
    {{"query_type": "deep_psychology", "query": ""}}
  ]
}}

Rules:
- Queries should be 8-18 words each.
- Use cricket-specific language.
- Include the cricket role where relevant.
- Do not make all three queries similar.
- Do not include quotation marks inside the query.
- Do not ask questions; write search phrases.
- Return JSON only."""


ANGLE_PROMPT = """You are an elite cricket psychology content strategist.

Your task is to choose the strongest content angle for a social media post or reel.

Original problem:
{problem}

Cricket role:
{cricket_role}

Player level:
{player_level}

Psychology angle:
{psychology_angle}

Content brief:
{content_brief}

Selected retrieved knowledge:
{selected_knowledge}

Generate 5 possible content angles, then choose the best one.

IMPORTANT: Each angle_options item MUST use EXACTLY these field names:
  "angle_name", "one_line_idea", "why_it_could_work", "risk"

The chosen_angle MUST use EXACTLY these field names:
  "angle_name", "core_message", "emotional_hook", "practical_promise", "why_this_is_best"

Return only valid JSON:

{{
  "angle_options": [
    {{"angle_name": "", "one_line_idea": "", "why_it_could_work": "", "risk": ""}}
  ],
  "chosen_angle": {{
    "angle_name": "",
    "core_message": "",
    "emotional_hook": "",
    "practical_promise": "",
    "why_this_is_best": ""
  }}
}}

Rules:
- Make the angle cricket-specific.
- Avoid generic motivation.
- Prefer angles that expose a hidden mental pattern.
- Prefer angles that give the player a practical shift.
- Do not use fake player stories.
- Use retrieved knowledge, but do not copy it word-for-word.
- Return JSON only."""


FINAL_PROMPT = """You are an elite cricket psychology content creator.

You create content for cricketers who want to improve their mental game.

Your content must be:
- cricket-specific
- emotionally sharp
- practical
- simple to understand
- grounded in retrieved sport psychology knowledge
- useful for players at the selected level
- written like a strong coach speaking directly to a player
- not generic motivation

Input:

Original problem:
{problem}

Category:
{category}

Sub-category:
{sub_category}

Cricket role:
{cricket_role}

Player level:
{player_level}

Psychology angle:
{psychology_angle}

Content angle:
{content_angle}

Content brief:
{content_brief_json}

Generated retrieval queries:
{queries_json}

Selected retrieved knowledge chunks:
{selected_chunks_json}

Chosen content angle:
{chosen_angle_json}

Create a complete content pack.

Return only valid JSON with this exact schema:

{{
  "topic": "",
  "content_title": "",
  "core_insight": "",
  "hook_options": ["", "", "", "", ""],
  "primary_hook": "",
  "reel_script": {{"opening": "", "problem": "", "reframe": "", "practical_advice": "", "closing": ""}},
  "short_reel_script": "",
  "caption": "",
  "carousel_slides": [
    {{"slide_number": 1, "slide_text": ""}},
    {{"slide_number": 2, "slide_text": ""}},
    {{"slide_number": 3, "slide_text": ""}},
    {{"slide_number": 4, "slide_text": ""}},
    {{"slide_number": 5, "slide_text": ""}},
    {{"slide_number": 6, "slide_text": ""}}
  ],
  "mental_drill": {{"name": "", "when_to_use": "", "steps": ["", "", ""]}},
  "one_line_takeaway": "",
  "cta": "",
  "hashtags": ["", "", "", "", ""],
  "source_grounding": {{"retrieved_chunk_ids_used": [], "key_ideas_used": []}}
}}

Writing rules:
- Speak directly to the cricketer using "you".
- Use cricket language naturally.
- Make the first 2 lines strong enough for Instagram.
- Avoid fake stories or made-up player examples.
- Do not mention Pinecone, RAG, retrieved chunks, or source metadata in the public-facing content.
- Do not sound like a textbook.
- Do not overuse words like mindset, confidence, champion, elite, unstoppable, or grind.
- Avoid generic lines like "believe in yourself" unless you explain exactly how.
- Give one practical mental action the player can actually try.
- The reel script should be suitable for a 30-60 second video.
- The caption should be readable on Instagram with short paragraphs.
- The carousel slides should be punchy and easy to read.
- Return JSON only."""

JSON_REPAIR_PROMPT = """The following JSON was returned by an AI but failed schema validation.

Validation error:
{error}

Invalid JSON:
{raw_json}

Fix ONLY the schema issues described in the error. Do not change any actual content values.
Return only valid JSON. No markdown fences. No explanation."""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """Strip markdown fences and parse JSON robustly."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        raw = "\n".join(inner).strip()
    return json.loads(raw)


def _llm_call(
    client: OpenAI,
    model: str,
    prompt: str,
    tracker: Optional[TokenTracker] = None,
    node_name: str = "unknown",
    self_heal_model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """Call the LLM, track tokens, and self-heal on JSON parse failure."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    if tracker and response.usage:
        tracker.record(node_name, model, response.usage)

    raw = response.choices[0].message.content
    try:
        return _parse_llm_json(raw)
    except json.JSONDecodeError as first_err:
        heal_response = client.chat.completions.create(
            model=self_heal_model,
            messages=[{"role": "user", "content": JSON_REPAIR_PROMPT.format(
                error=str(first_err),
                raw_json=raw,
            )}],
            temperature=0.0,
        )
        if tracker and heal_response.usage:
            tracker.record(f"{node_name}:self_heal", self_heal_model, heal_response.usage)
        healed_raw = heal_response.choices[0].message.content
        try:
            return _parse_llm_json(healed_raw)
        except json.JSONDecodeError as second_err:
            raise json.JSONDecodeError(
                f"LLM JSON failed after self-heal. Original: {first_err}. Healed: {second_err}",
                second_err.doc,
                second_err.pos,
            )


def _embed_text(
    client: OpenAI,
    model: str,
    text: str,
    tracker: Optional[TokenTracker] = None,
    node_name: str = "retrieval",
) -> List[float]:
    result = client.embeddings.create(model=model, input=text)
    if tracker and result.usage:
        tracker.record_embedding(node_name, model, result.usage.total_tokens)
    return result.data[0].embedding


def _clean_chunk(match: Any, query_type: str) -> Dict[str, Any]:
    metadata = getattr(match, "metadata", {}) or {}
    text = metadata.get("text", "")
    chunk_id = (
        getattr(match, "id", None)
        or metadata.get("_id")
        or hashlib.md5(text.encode("utf-8")).hexdigest()
    )
    return {
        "id": chunk_id,
        "score": float(getattr(match, "score", 0.0) or 0.0),
        "query_type": query_type,
        "title": metadata.get("title", ""),
        "text": text[:2200],
        "concept": metadata.get("concept", ""),
        "concepts": metadata.get("concepts", []),
        "topic": metadata.get("topic", ""),
        "topics": metadata.get("topics", []),
        "cricket_role": metadata.get("cricket_role", ""),
        "source": metadata.get("source", ""),
        "author": metadata.get("author", ""),
        "source_priority": metadata.get("source_priority", 0),
        "relevance_score": metadata.get("relevance_score", 0),
        "content_type": metadata.get("content_type", ""),
    }


# ---------------------------------------------------------------------------
# Chunk selection (15 raw → 9 selected, diversity-aware)
# ---------------------------------------------------------------------------

def _dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_ids: set = set()
    seen_text: set = set()
    deduped = []
    for chunk in chunks:
        text_key = hashlib.md5(
            chunk.get("text", "").strip().lower().encode("utf-8")
        ).hexdigest()
        if chunk["id"] in seen_ids or text_key in seen_text:
            continue
        seen_ids.add(chunk["id"])
        seen_text.add(text_key)
        deduped.append(chunk)
    return deduped


def _normalize_and_rank(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    diversity_bonus_map = {
        "symptom": 0.04,
        "mental_skill": 0.06,
        "deep_psychology": 0.08,
    }
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        by_type.setdefault(c["query_type"], []).append(c)

    for query_type, group in by_type.items():
        scores = [g["score"] for g in group]
        max_s, min_s = max(scores), min(scores)
        for g in group:
            normalized = (
                1.0 if max_s == min_s else (g["score"] - min_s) / (max_s - min_s)
            )
            priority_bonus = min(float(g.get("source_priority", 0)) / 100.0, 0.10)
            div_bonus = diversity_bonus_map.get(query_type, 0.04)
            g["normalized_score"] = normalized
            g["final_rank_score"] = (
                0.75 * normalized + 0.15 * div_bonus + 0.10 * priority_bonus
            )
    return chunks


def select_top_chunks(
    chunks: List[Dict[str, Any]], final_chunk_count: int = 9
) -> List[Dict[str, Any]]:
    deduped = _dedupe_chunks(chunks)
    ranked = _normalize_and_rank(deduped)

    by_type = {
        qt: sorted(
            [c for c in ranked if c["query_type"] == qt],
            key=lambda x: x["final_rank_score"],
            reverse=True,
        )
        for qt in ["symptom", "mental_skill", "deep_psychology"]
    }

    selected: List[Dict[str, Any]] = []
    selected_ids: set = set()

    for qt in ["symptom", "mental_skill", "deep_psychology"]:
        for chunk in by_type.get(qt, [])[:2]:
            if chunk["id"] not in selected_ids:
                selected.append(chunk)
                selected_ids.add(chunk["id"])

    remaining = sorted(
        [c for c in ranked if c["id"] not in selected_ids],
        key=lambda x: x["final_rank_score"],
        reverse=True,
    )
    for chunk in remaining:
        if len(selected) >= final_chunk_count:
            break
        selected.append(chunk)
        selected_ids.add(chunk["id"])

    return sorted(selected, key=lambda x: x["final_rank_score"], reverse=True)[
        :final_chunk_count
    ]


# ---------------------------------------------------------------------------
# HTML output renderer
# ---------------------------------------------------------------------------

def _render_html(
    data: Dict[str, Any],
    payload: Dict[str, Any],
    tracker: Optional[TokenTracker] = None,
) -> str:
    fc = data.get("final_content", {})

    def esc(s: Any) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    hook_items = "".join(f"<li>{esc(h)}</li>" for h in fc.get("hook_options", []))

    reel = fc.get("reel_script", {})
    reel_rows = "".join(
        f"<tr><td class='label'>{esc(k.replace('_',' ').title())}</td><td>{esc(v)}</td></tr>"
        for k, v in reel.items()
    )

    slides = fc.get("carousel_slides", [])
    slide_cards = "".join(
        f"<div class='slide-card'><span class='slide-num'>Slide {esc(s.get('slide_number',''))}</span><p>{esc(s.get('slide_text',''))}</p></div>"
        for s in slides
    )

    drill = fc.get("mental_drill", {})
    drill_steps = "".join(f"<li>{esc(step)}</li>" for step in drill.get("steps", []))

    hashtags = " ".join(f"<span class='tag'>{esc(h)}</span>" for h in fc.get("hashtags", []))

    sg = fc.get("source_grounding", {})
    chunk_ids = ", ".join(esc(c) for c in sg.get("retrieved_chunk_ids_used", []))
    key_ideas = "".join(f"<li>{esc(k)}</li>" for k in sg.get("key_ideas_used", []))

    # ── Token / cost section ────────────────────────────────────────────────
    usage_section = ""
    if tracker:
        totals = tracker.totals()
        node_rows = "".join(
            f"""
            <tr>
              <td class='label'>{esc(e.node)}</td>
              <td>{esc(e.model)}</td>
              <td class='num'>{e.prompt_tokens:,}</td>
              <td class='num'>{e.completion_tokens:,}</td>
              <td class='num'>{e.total_tokens:,}</td>
              <td class='num cost'>${e.cost_usd:.6f}</td>
            </tr>"""
            for e in tracker.entries
        )
        usage_section = f"""
        <section class='usage-section'>
          <h2>💰 Token Usage &amp; Cost</h2>
          <div class='usage-totals'>
            <div class='stat-box'>
              <span class='stat-label'>Total Tokens</span>
              <span class='stat-value'>{totals['total_tokens']:,}</span>
            </div>
            <div class='stat-box'>
              <span class='stat-label'>Prompt Tokens</span>
              <span class='stat-value'>{totals['total_prompt_tokens']:,}</span>
            </div>
            <div class='stat-box'>
              <span class='stat-label'>Completion Tokens</span>
              <span class='stat-value'>{totals['total_completion_tokens']:,}</span>
            </div>
            <div class='stat-box highlight'>
              <span class='stat-label'>Total Cost</span>
              <span class='stat-value'>${totals['total_cost_usd']:.6f}</span>
            </div>
            <div class='stat-box'>
              <span class='stat-label'>Elapsed Time</span>
              <span class='stat-value'>{totals['elapsed_seconds']}s</span>
            </div>
          </div>
          <h3>Per-Node Breakdown</h3>
          <table class='usage-table'>
            <thead>
              <tr>
                <th>Node</th><th>Model</th>
                <th>Prompt</th><th>Completion</th><th>Total</th><th>Cost (USD)</th>
              </tr>
            </thead>
            <tbody>{node_rows}</tbody>
          </table>
          <p class='muted'>Prices: gpt-4.1 $2.00/$8.00 · gpt-4.1-mini $0.40/$1.60 · text-embedding-3-small $0.02 per 1M tokens</p>
        </section>"""

    # ── Debug section ───────────────────────────────────────────────────────
    debug_section = ""
    if data.get("content_brief"):
        brief_rows = "".join(
            f"<tr><td class='label'>{esc(k)}</td><td>{esc(v)}</td></tr>"
            for k, v in data["content_brief"].items()
        )
        angle = data.get("angle_selection", {})
        chosen = angle.get("chosen_angle", {})
        chosen_rows = "".join(
            f"<tr><td class='label'>{esc(k)}</td><td>{esc(v)}</td></tr>"
            for k, v in chosen.items()
        )
        queries = data.get("retrieval_queries", [])
        query_rows = "".join(
            f"<tr><td class='label'>{esc(q.get('query_type',''))}</td><td>{esc(q.get('query',''))}</td></tr>"
            for q in queries
        )
        debug_section = f"""
        <section class='debug-section'>
          <h2>🛠 Debug Info</h2>
          <h3>Content Brief</h3>
          <table>{brief_rows}</table>
          <h3>Retrieval Queries</h3>
          <table>{query_rows}</table>
          <h3>Chosen Angle</h3>
          <table>{chosen_rows}</table>
          <p class='muted'>Retrieved candidate chunks: {esc(data.get('retrieved_candidate_count', 0))}</p>
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{esc(fc.get('content_title', 'Content Pack'))}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      padding: 2rem 1rem;
      line-height: 1.6;
    }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    h1 {{ font-size: 1.8rem; color: #f8fafc; margin-bottom: .25rem; }}
    h2 {{ font-size: 1.2rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .08em; margin: 2rem 0 .75rem; border-bottom: 1px solid #1e293b; padding-bottom: .5rem; }}
    h3 {{ font-size: 1rem; color: #64748b; margin: 1.25rem 0 .5rem; }}
    section {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
    .meta {{ color: #64748b; font-size: .85rem; margin-bottom: 1rem; }}
    p {{ margin-bottom: .75rem; color: #cbd5e1; }}
    ul {{ padding-left: 1.25rem; color: #cbd5e1; }}
    ul li {{ margin-bottom: .4rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: .5rem; }}
    td, th {{ padding: .5rem .75rem; vertical-align: top; border-bottom: 1px solid #0f172a; font-size: .9rem; text-align: left; }}
    th {{ color: #64748b; font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }}
    td.label {{ width: 170px; color: #64748b; font-weight: 600; white-space: nowrap; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    td.cost {{ color: #34d399; font-weight: 600; }}
    .primary-hook {{ background: #0f2a47; border-left: 4px solid #3b82f6; padding: 1rem 1.25rem; border-radius: 8px; font-size: 1.05rem; color: #bfdbfe; margin-bottom: 1rem; }}
    .short-reel {{ background: #0f172a; border-radius: 8px; padding: 1rem; white-space: pre-wrap; font-size: .92rem; color: #e2e8f0; }}
    .caption-box {{ background: #0f172a; border-radius: 8px; padding: 1rem; white-space: pre-wrap; font-size: .92rem; color: #e2e8f0; }}
    .slides-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; margin-top: .5rem; }}
    .slide-card {{ background: #0f172a; border-radius: 10px; padding: 1rem; }}
    .slide-num {{ font-size: .75rem; color: #64748b; display: block; margin-bottom: .35rem; text-transform: uppercase; letter-spacing: .05em; }}
    .slide-card p {{ color: #cbd5e1; font-size: .88rem; margin: 0; }}
    .tag {{ display: inline-block; background: #0c2a1a; border: 1px solid #166534; color: #4ade80; border-radius: 999px; padding: .2rem .75rem; font-size: .8rem; margin: .15rem; }}
    .takeaway {{ font-size: 1.1rem; font-weight: 600; color: #f8fafc; padding: 1rem; background: #16213e; border-left: 4px solid #6366f1; border-radius: 8px; }}
    .cta-box {{ background: #1a0533; border-left: 4px solid #a855f7; padding: 1rem; border-radius: 8px; color: #e9d5ff; }}
    .muted {{ color: #475569; font-size: .8rem; margin-top: .75rem; }}
    .usage-section {{ background: #0f2a1a; border: 1px solid #166534; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
    .usage-section h2 {{ color: #4ade80; border-bottom-color: #166534; }}
    .usage-totals {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: .75rem; margin-bottom: 1.25rem; }}
    .stat-box {{ background: #0f172a; border-radius: 8px; padding: .75rem 1rem; }}
    .stat-box.highlight {{ background: #064e3b; border: 1px solid #34d399; }}
    .stat-label {{ display: block; font-size: .72rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; margin-bottom: .2rem; }}
    .stat-value {{ display: block; font-size: 1.15rem; font-weight: 700; color: #f8fafc; font-variant-numeric: tabular-nums; }}
    .stat-box.highlight .stat-value {{ color: #34d399; }}
    .usage-table th {{ background: #0f172a; }}
    .debug-section {{ background: #1a1a2e; border: 1px dashed #334155; border-radius: 12px; padding: 1.5rem; margin-top: 2rem; }}
    .debug-section h2 {{ color: #f59e0b; border-bottom-color: #334155; }}
    .badge {{ display: inline-block; background: #1e3a2f; color: #34d399; padding: .15rem .6rem; border-radius: 6px; font-size: .75rem; font-weight: 700; margin-left: .5rem; }}
    @media (max-width: 600px) {{ td.label {{ width: 100px; }} .usage-totals {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
<div class="container">
  <section>
    <h1>{esc(fc.get('content_title', ''))} <span class='badge'>RAG Output</span></h1>
    <div class="meta">
      {esc(payload.get('cricket_role',''))} &nbsp;·&nbsp;
      {esc(payload.get('player_level',''))} &nbsp;·&nbsp;
      {esc(payload.get('category',''))} → {esc(payload.get('sub_category',''))} &nbsp;·&nbsp;
      Angle: {esc(payload.get('psychology_angle',''))}
    </div>
    <p><strong>Topic:</strong> {esc(fc.get('topic', ''))}</p>
    <p><strong>Core Insight:</strong> {esc(fc.get('core_insight', ''))}</p>
  </section>

  {usage_section}

  <section>
    <h2>🎣 Hooks</h2>
    <div class="primary-hook">⭐ {esc(fc.get('primary_hook', ''))}</div>
    <ul>{hook_items}</ul>
  </section>

  <section>
    <h2>🎬 Reel Script</h2>
    <table>{reel_rows}</table>
    <h3>Short Reel Script</h3>
    <div class="short-reel">{esc(fc.get('short_reel_script', ''))}</div>
  </section>

  <section>
    <h2>📝 Caption</h2>
    <div class="caption-box">{esc(fc.get('caption', ''))}</div>
  </section>

  <section>
    <h2>🎠 Carousel Slides</h2>
    <div class="slides-grid">{slide_cards}</div>
  </section>

  <section>
    <h2>🧠 Mental Drill</h2>
    <p><strong>{esc(drill.get('name', ''))}</strong> — {esc(drill.get('when_to_use', ''))}</p>
    <ul>{drill_steps}</ul>
  </section>

  <section>
    <h2>💡 Takeaway &amp; CTA</h2>
    <div class="takeaway">{esc(fc.get('one_line_takeaway', ''))}</div>
    <div class="cta-box" style="margin-top:1rem">{esc(fc.get('cta', ''))}</div>
  </section>

  <section>
    <h2>#️⃣ Hashtags</h2>
    <div>{hashtags}</div>
  </section>

  <section>
    <h2>📚 Source Grounding</h2>
    <p class="muted">Chunk IDs used: {chunk_ids}</p>
    <ul>{key_ideas}</ul>
  </section>

  {debug_section}
</div>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# App-level shared state
# ---------------------------------------------------------------------------

class AppState:
    settings: Settings
    openai_client: OpenAI
    pinecone_client: Pinecone
    pinecone_index: Any
    graph: Any
    checkpointer: Any


_app_state_ref: AppState = None


def _get_services() -> AppState:
    return _app_state_ref


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def node_run_brief(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
    payload = state["payload"]
    tracker: TokenTracker = state.get("tracker")

    summary = state.get("brief_summary", "")
    summary_context = (
        f"Context from your previous brief for a related request:\n{summary}\n"
        if summary else ""
    )

    raw = _llm_call(
        svc.openai_client,
        svc.settings.brief_model,
        BRIEF_PROMPT.format(
            summary_context=summary_context,
            problem=payload["problem"],
            category=payload["category"],
            sub_category=payload["sub_category"],
            cricket_role=payload["cricket_role"],
            player_level=payload["player_level"],
            psychology_angle=payload["psychology_angle"],
        ),
        tracker=tracker,
        node_name="run_brief",
    )
    brief = BriefResponse.model_validate(raw).model_dump()
    new_messages = [
        HumanMessage(content=f"Problem: {payload['problem']}"),
        AIMessage(content=json.dumps(brief)),
    ]
    return {"brief": brief, "messages": new_messages}


def node_run_memory_summary(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
    tracker: TokenTracker = state.get("tracker")
    messages = state.get("messages", [])

    if len(messages) < 2:
        return {}

    existing_summary = state.get("brief_summary", "")
    if existing_summary:
        summary_prompt = (
            f"Existing summary: {existing_summary}\n\n"
            f"Extend this summary with the latest exchange:\n"
            f"User: {messages[-2].content}\nAssistant: {messages[-1].content}"
        )
    else:
        summary_prompt = (
            f"Summarise this brief generation exchange in 2-3 sentences:\n"
            f"User: {messages[-2].content}\nAssistant: {messages[-1].content}"
        )

    response = svc.openai_client.chat.completions.create(
        model=svc.settings.summary_model,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.2,
        max_tokens=200,
    )
    if tracker and response.usage:
        tracker.record("run_memory_summary", svc.settings.summary_model, response.usage)

    new_summary = response.choices[0].message.content.strip()
    delete_msgs = [RemoveMessage(id=m.id) for m in messages[:-2]]
    return {"brief_summary": new_summary, "messages": delete_msgs}


def node_run_queries(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
    payload = state["payload"]
    tracker: TokenTracker = state.get("tracker")

    raw = _llm_call(
        svc.openai_client,
        svc.settings.query_model,
        QUERY_PROMPT.format(
            problem=payload["problem"],
            cricket_role=payload["cricket_role"],
            player_level=payload["player_level"],
            psychology_angle=payload["psychology_angle"],
            category=payload["category"],
            content_brief=json.dumps(state["brief"], indent=2),
        ),
        tracker=tracker,
        node_name="run_queries",
    )
    queries = QueryGenerationResponse.model_validate(raw).model_dump()["queries"]
    return {"queries": queries}


def node_run_retrieval(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
    tracker: TokenTracker = state.get("tracker")

    all_chunks: List[Dict[str, Any]] = []
    for q in state["queries"]:
        vector = _embed_text(
            svc.openai_client,
            svc.settings.embedding_model,
            q["query"],
            tracker=tracker,
            node_name=f"retrieval:{q['query_type']}",
        )
        result = svc.pinecone_index.query(
            namespace=svc.settings.pinecone_namespace,
            vector=vector,
            top_k=svc.settings.top_k_per_query,
            include_metadata=True,
        )
        for match in getattr(result, "matches", []) or []:
            all_chunks.append(_clean_chunk(match, q["query_type"]))

    return {"raw_chunks": all_chunks}


def node_run_chunk_selection(state: PipelineState) -> Dict[str, Any]:
    selected = select_top_chunks(
        state["raw_chunks"],
        state["payload"].get("final_chunk_count", 9),
    )
    return {"selected_chunks": selected}


def node_run_angle_selection(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
    payload = state["payload"]
    tracker: TokenTracker = state.get("tracker")

    raw = _llm_call(
        svc.openai_client,
        svc.settings.angle_model,
        ANGLE_PROMPT.format(
            problem=payload["problem"],
            cricket_role=payload["cricket_role"],
            player_level=payload["player_level"],
            psychology_angle=payload["psychology_angle"],
            content_brief=json.dumps(state["brief"], indent=2),
            selected_knowledge=json.dumps(state["selected_chunks"], indent=2),
        ),
        tracker=tracker,
        node_name="run_angle_selection",
    )
    angle_selection = AngleSelectionResponse.model_validate(raw).model_dump()
    return {"angle_selection": angle_selection}


def node_run_final_content(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
    payload = state["payload"]
    tracker: TokenTracker = state.get("tracker")

    raw = _llm_call(
        svc.openai_client,
        svc.settings.final_model,
        FINAL_PROMPT.format(
            problem=payload["problem"],
            category=payload["category"],
            sub_category=payload["sub_category"],
            cricket_role=payload["cricket_role"],
            player_level=payload["player_level"],
            psychology_angle=payload["psychology_angle"],
            content_angle=state["angle_selection"]["chosen_angle"]["angle_name"],
            content_brief_json=json.dumps(state["brief"], indent=2),
            queries_json=json.dumps(state["queries"], indent=2),
            selected_chunks_json=json.dumps(state["selected_chunks"], indent=2),
            chosen_angle_json=json.dumps(state["angle_selection"], indent=2),
        ),
        tracker=tracker,
        node_name="run_final_content",
    )
    final_content = FinalContentResponse.model_validate(raw).model_dump()
    return {"final_content": final_content}


# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(PipelineState)

    builder.add_node("run_brief", node_run_brief)
    builder.add_node("run_memory_summary", node_run_memory_summary)
    builder.add_node("run_queries", node_run_queries)
    builder.add_node("run_retrieval", node_run_retrieval)
    builder.add_node("run_chunk_selection", node_run_chunk_selection)
    builder.add_node("run_angle_selection", node_run_angle_selection)
    builder.add_node("run_final_content", node_run_final_content)

    builder.add_edge(START, "run_brief")
    builder.add_edge("run_brief", "run_memory_summary")
    builder.add_edge("run_memory_summary", "run_queries")
    builder.add_edge("run_queries", "run_retrieval")
    builder.add_edge("run_retrieval", "run_chunk_selection")
    builder.add_edge("run_chunk_selection", "run_angle_selection")
    builder.add_edge("run_angle_selection", "run_final_content")
    builder.add_edge("run_final_content", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer), checkpointer


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_state_ref
    from dotenv import load_dotenv
    load_dotenv()

    settings = Settings.from_env()
    state = AppState()
    state.settings = settings
    state.openai_client = OpenAI(api_key=settings.openai_api_key)
    state.pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    state.pinecone_index = state.pinecone_client.Index(host=settings.pinecone_index_host)
    state.graph, state.checkpointer = build_graph()
    _app_state_ref = state
    app.state.services = state
    yield


app = FastAPI(
    title="Cricket Psychology RAG API",
    version="3.1.0",
    description="LangGraph 7-node RAG pipeline for cricket psychology content generation.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/generate-content",
    response_class=HTMLResponse,
    dependencies=[Depends(require_api_key)],
)
def generate_content(payload: GenerateContentRequest):
    try:
        svc: AppState = app.state.services
        config = {"configurable": {"thread_id": payload.thread_id}}
        tracker = TokenTracker()
        t_start = time.monotonic()

        initial_state: PipelineState = {
            "payload": payload.model_dump(),
            "messages": [],
            "brief_summary": "",
            "brief": None,
            "queries": None,
            "raw_chunks": [],
            "selected_chunks": None,
            "angle_selection": None,
            "final_content": None,
            "tracker": tracker,
        }

        result = svc.graph.invoke(initial_state, config=config)
        tracker.elapsed_seconds = time.monotonic() - t_start

        response_data: Dict[str, Any] = {
            "status": "success",
            "input": payload.model_dump(),
            "final_content": result["final_content"],
        }
        if payload.debug:
            response_data.update({
                "content_brief": result["brief"],
                "retrieval_queries": result["queries"],
                "retrieved_candidate_count": len(result["raw_chunks"]),
                "selected_chunks": result["selected_chunks"],
                "angle_selection": result["angle_selection"],
                "brief_summary_used": result.get("brief_summary", ""),
            })

        return HTMLResponse(
            content=_render_html(response_data, payload.model_dump(), tracker=tracker)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON after self-heal: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
