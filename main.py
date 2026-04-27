import json
import os
import hashlib
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict, List, Literal, Optional
from operator import add

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from pinecone import Pinecone

from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_host: str
    pinecone_namespace: str = "content-automate"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
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
            llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            top_k_per_query=int(os.getenv("TOP_K_PER_QUERY", "5")),
            final_chunk_count=int(os.getenv("FINAL_CHUNK_COUNT", "9")),
        )


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


class AngleOption(BaseModel):
    angle_name: str
    one_line_idea: str
    why_it_could_work: str
    risk: str


class ChosenAngle(BaseModel):
    angle_name: str
    core_message: str
    emotional_hook: str
    practical_promise: str
    why_this_is_best: str


class AngleSelectionResponse(BaseModel):
    angle_options: List[AngleOption]
    chosen_angle: ChosenAngle


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
    # ── Inputs ──────────────────────────────────────────────────────────────
    payload: Dict[str, Any]

    # ── Memory: only lives in brief_node (Node 1) ───────────────────────────
    # Full message history managed by add_messages reducer.
    # After each run the summarize_brief_memory node trims to last 2 messages
    # and stores a rolling summary string.
    messages: Annotated[List[Any], add_messages]
    brief_summary: str                    # rolling summary of prior brief calls

    # ── Pipeline stage outputs ───────────────────────────────────────────────
    brief: Optional[Dict[str, Any]]
    queries: Optional[List[Dict[str, Any]]]
    raw_chunks: Annotated[List[Dict[str, Any]], add]   # accumulates via reducer
    selected_chunks: Optional[List[Dict[str, Any]]]
    angle_selection: Optional[Dict[str, Any]]
    final_content: Optional[Dict[str, Any]]


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


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """Strip markdown fences and parse JSON robustly."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _llm_call(client: OpenAI, model: str, prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        timeout=60,
    )
    return _parse_llm_json(response.choices[0].message.content)


def _embed_text(client: OpenAI, model: str, text: str) -> List[float]:
    result = client.embeddings.create(model=model, input=text)
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
    """
    Compute final_rank_score per chunk.
    Formula: 0.75 * normalized_vector_score
           + 0.15 * diversity_bonus   (varies by query_type)
           + 0.10 * source_priority   (capped at 0.10)
    """
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
    """
    Dedup → rank → guarantee diversity (2 per query_type first)
    → fill remainder by score → cap at final_chunk_count.
    Operates on the accumulated raw_chunks from PipelineState.
    """
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

    # Guarantee at least 2 from each query type
    for qt in ["symptom", "mental_skill", "deep_psychology"]:
        for chunk in by_type.get(qt, [])[:2]:
            if chunk["id"] not in selected_ids:
                selected.append(chunk)
                selected_ids.add(chunk["id"])

    # Fill remaining slots with highest ranked remaining chunks
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
# App-level shared state
# ---------------------------------------------------------------------------

class AppState:
    settings: Settings
    openai_client: OpenAI
    pinecone_client: Pinecone
    pinecone_index: Any
    graph: Any
    checkpointer: Any


_app_state_ref: AppState = None  # set during lifespan


def _get_services() -> AppState:
    return _app_state_ref


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

# Node 1 – Brief generation (only node that uses memory)
def node_brief(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
    payload = state["payload"]

    summary = state.get("brief_summary", "")
    summary_context = (
        f"Context from your previous brief for a related request:\n{summary}\n"
        if summary
        else ""
    )

    raw = _llm_call(
        svc.openai_client,
        svc.settings.llm_model,
        BRIEF_PROMPT.format(
            summary_context=summary_context,
            problem=payload["problem"],
            category=payload["category"],
            sub_category=payload["sub_category"],
            cricket_role=payload["cricket_role"],
            player_level=payload["player_level"],
            psychology_angle=payload["psychology_angle"],
        ),
    )
    brief = BriefResponse.model_validate(raw).model_dump()

    # Record this turn: HumanMessage = input, AIMessage = brief output
    new_messages = [
        HumanMessage(content=f"Problem: {payload['problem']}"),
        AIMessage(content=json.dumps(brief)),
    ]
    return {"brief": brief, "messages": new_messages}


# Node 1b – Summarize & trim brief memory (keeps last 1 pair = 2 messages)
def node_summarize_brief_memory(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
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
        model=svc.settings.llm_model,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.2,
        max_tokens=200,
        timeout=30,
    )
    new_summary = response.choices[0].message.content.strip()

    # Delete all messages except the last 2 (most recent Human+AI pair)
    delete_msgs = [RemoveMessage(id=m.id) for m in messages[:-2]]
    return {"brief_summary": new_summary, "messages": delete_msgs}


# Node 2 – Query generation
def node_generate_queries(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
    payload = state["payload"]
    brief = state["brief"]

    raw = _llm_call(
        svc.openai_client,
        svc.settings.llm_model,
        QUERY_PROMPT.format(
            problem=payload["problem"],
            cricket_role=payload["cricket_role"],
            player_level=payload["player_level"],
            psychology_angle=payload["psychology_angle"],
            category=payload["category"],
            content_brief=json.dumps(brief, indent=2),
        ),
    )
    queries = QueryGenerationResponse.model_validate(raw).model_dump()["queries"]
    return {"queries": queries}


# Node 3 – Retrieve chunks from Pinecone (all 3 queries)
def node_retrieve_chunks(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
    queries = state["queries"]

    all_chunks: List[Dict[str, Any]] = []
    for q in queries:
        vector = _embed_text(
            svc.openai_client, svc.settings.embedding_model, q["query"]
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


# Node 4 – Select top-N chunks (15 → 9, diversity-aware)
def node_select_chunks(state: PipelineState) -> Dict[str, Any]:
    selected = select_top_chunks(
        state["raw_chunks"],
        state["payload"].get("final_chunk_count", 9),
    )
    return {"selected_chunks": selected}


# Node 5 – Angle selection
def node_select_angle(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
    payload = state["payload"]

    raw = _llm_call(
        svc.openai_client,
        svc.settings.llm_model,
        ANGLE_PROMPT.format(
            problem=payload["problem"],
            cricket_role=payload["cricket_role"],
            player_level=payload["player_level"],
            psychology_angle=payload["psychology_angle"],
            content_brief=json.dumps(state["brief"], indent=2),
            selected_knowledge=json.dumps(state["selected_chunks"], indent=2),
        ),
    )
    angle_selection = AngleSelectionResponse.model_validate(raw).model_dump()
    return {"angle_selection": angle_selection}


# Node 6 – Final content generation
def node_generate_content(state: PipelineState) -> Dict[str, Any]:
    svc = _get_services()
    payload = state["payload"]

    raw = _llm_call(
        svc.openai_client,
        svc.settings.llm_model,
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
    )
    final_content = FinalContentResponse.model_validate(raw).model_dump()
    return {"final_content": final_content}


# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(PipelineState)

    builder.add_node("brief", node_brief)
    builder.add_node("summarize_brief_memory", node_summarize_brief_memory)
    builder.add_node("generate_queries", node_generate_queries)
    builder.add_node("retrieve_chunks", node_retrieve_chunks)
    builder.add_node("select_chunks", node_select_chunks)
    builder.add_node("select_angle", node_select_angle)
    builder.add_node("generate_content", node_generate_content)

    builder.add_edge(START, "brief")
    builder.add_edge("brief", "summarize_brief_memory")
    builder.add_edge("summarize_brief_memory", "generate_queries")
    builder.add_edge("generate_queries", "retrieve_chunks")
    builder.add_edge("retrieve_chunks", "select_chunks")
    builder.add_edge("select_chunks", "select_angle")
    builder.add_edge("select_angle", "generate_content")
    builder.add_edge("generate_content", END)

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
    version="2.0.0",
    description="LangGraph 6-node RAG pipeline for cricket psychology content generation.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-content")
def generate_content(payload: GenerateContentRequest):
    try:
        svc: AppState = app.state.services
        config = {"configurable": {"thread_id": payload.thread_id}}

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
        }

        result = svc.graph.invoke(initial_state, config=config)

        response: Dict[str, Any] = {
            "status": "success",
            "input": payload.model_dump(),
            "final_content": result["final_content"],
        }
        if payload.debug:
            response.update({
                "content_brief": result["brief"],
                "retrieval_queries": result["queries"],
                "retrieved_candidate_count": len(result["raw_chunks"]),
                "selected_chunks": result["selected_chunks"],
                "angle_selection": result["angle_selection"],
                "brief_summary_used": result.get("brief_summary", ""),
            })
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
