"""
Generic RAG router workflow with human-in-the-loop clarify (LlamaIndex Workflows).

Differences from the turn-boundary version:
  * Clarify is in-run HITL. The `clarify` step pauses via `ctx.wait_for_event`,
    emits the question as an InputRequiredEvent, and resumes the SAME run when
    the caller sends back a HumanResponseEvent. No `pending_clarification` flag,
    no re-routing. The clarified query flows into retrieve -> synth -> evaluate,
    so it gets scored too.
  * Router routes bare topic words ("татвар") to clarify, not retrieve, using
    explicit intent definitions + few-shot examples.
  * Clarify questions are topic-aware: the detected topic's known sub-areas are
    offered as concrete options.

Retrieval stays a single injected callable: `retriever(query, topics) -> context`.
Put your hybrid (dense+sparse BGE-M3) + reranker pipeline there. `topics` is an
opaque filter LIST the workflow forwards but never interprets (empty = no filter).

Deployment note: a paused HITL run stays alive while waiting, so it counts
against the workflow `timeout`. For a streaming session (CLI / WebSocket / SSE)
this is fine — set a generous timeout. For stateless HTTP, snapshot the context
(`ctx.to_dict()`) when InputRequiredEvent fires and rehydrate
(`Context.from_dict`) on the next request before sending HumanResponseEvent.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel

from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import (
    Context,
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI

from ema_agent.agent_starter import Topic
from ema_agent.prompts import (
    CLARIFY_FALLBACK as _CLARIFY_FALLBACK,
    CLARIFY_TMPL as _CLARIFY_TMPL,
    DECOMPOSE_TMPL as _DECOMPOSE_TMPL,
    DEFAULT_SYSTEM_PROMPT as _DEFAULT_SYSTEM_PROMPT,
    EVALUATE_TMPL as _EVALUATE_TMPL,
    NO_INFO as _NO_INFO,
    ROUTE_TMPL as _ROUTE_TMPL,
    SYNTH_TMPL as _SYNTH_TMPL,
    TOPIC_OPTIONS as _TOPIC_OPTIONS,
)

MAX_SUBQUERIES = 3

# A retriever is any async callable: (query, topics) -> context string.
# This is the ONLY integration point. Put hybrid retrieval + reranking here.
# `topics` is an opaque filter LIST the workflow forwards but never interprets;
# pass an empty list to disable filtering.
RetrieverFn = Callable[[str, list[str]], Awaitable[str]]


# ── Events ────────────────────────────────────────────────────────────────────

class StreamEvent(Event):
    delta: str

class ChatEvent(Event):
    query: str

class ClarifyEvent(Event):
    original_query: str
    topics: list[str] = []

class RetrieveEvent(Event):
    query: str
    topics: list[str] = []
    retry: int = 0

class ComplexEvent(Event):
    query: str
    topics: list[str] = []

class SubQueryEvent(Event):
    query: str
    topics: list[str]
    index: int

class RetrievalResultEvent(Event):
    context: str
    index: int

class AnswerEvent(Event):
    query: str
    context: str
    answer: str


# ── Structured output schemas ─────────────────────────────────────────────────

class RouteDecision(BaseModel):
    intent: Literal["chat", "clarify", "retrieve", "complex"]
    # The one domain-specific field: an optional retrieval filter LIST the
    # workflow forwards verbatim to your retriever without interpreting it.
    # The allowed values come from the `Topic` enum (the single source of truth
    # for the taxonomy, shared with the retriever's data-validation). Edit the
    # taxonomy there, not here. Empty list = no topic applies (NONE is a
    # retriever sentinel, not a router choice — don't emit it).
    topics: list[Topic] = []


class DecomposedQueries(BaseModel):
    sub_queries: list[str]


class AnswerEvaluation(BaseModel):
    """LLM grade of a retrieval-grounded answer."""
    score: int                                              # 1-10
    verdict: Literal["good", "weak_retrieval", "unanswerable"]
    reason: str = ""


# ── Workflow ──────────────────────────────────────────────────────────────────

class RouterWorkflow(Workflow):
    def __init__(
        self,
        *args: Any,
        retriever: RetrieverFn,
        llm: FunctionCallingLLM | None = None,
        system_prompt: str | None = None,
        route_prompt: PromptTemplate | None = None,
        route_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.retriever = retriever
        self.llm = llm or OpenAI(model="gpt-4o-mini", temperature=0.1)
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        # Route prompt + decision schema are injectable so each app picks its own
        # taxonomy. Defaults are the e-business FAQ pair; the decision MUST expose
        # `.intent` (chat|clarify|retrieve|complex) and `.topics` (opaque filter
        # list forwarded verbatim to the retriever).
        self.route_prompt = route_prompt or _ROUTE_TMPL
        self.route_schema = route_schema or RouteDecision

    # ── helpers ────────────────────────────────────────────────────────────────

    async def _get_memory(self, ctx: Context) -> ChatMemoryBuffer:
        memory = await ctx.store.get("memory", default=None)
        if memory is None:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)
            memory.put(ChatMessage(role="system", content=self.system_prompt))
        return memory

    # ── step 1: route ──────────────────────────────────────────────────────────

    @step
    async def route_query(
        self, ctx: Context, ev: StartEvent
    ) -> ChatEvent | ClarifyEvent | RetrieveEvent | ComplexEvent:
        query: str = ev.input

        memory = await self._get_memory(ctx)
        memory.put(ChatMessage(role="user", content=query))
        await ctx.store.set("memory", memory)
        await ctx.store.set("original_query", query)

        history = "\n".join(
            f"{m.role}: {m.content}" for m in memory.get()[-4:] if m.role != "system"
        )
        decision = await self.llm.astructured_predict(
            self.route_schema, self.route_prompt, query=query, history=history
        )
        print(f"[route] intent={decision.intent} topics={decision.topics}")

        if decision.intent == "chat":
            return ChatEvent(query=query)
        if decision.intent == "clarify":
            # Hand off to the HITL clarify step; it generates the question,
            # pauses for the human reply, then resumes into retrieval.
            return ClarifyEvent(original_query=query, topics=decision.topics)
        if decision.intent == "complex":
            return ComplexEvent(query=query, topics=decision.topics)
        return RetrieveEvent(query=query, topics=decision.topics, retry=0)

    # ── step 2a: chitchat (no retrieval, no scoring) ───────────────────────────

    @step
    async def handle_chat(self, ctx: Context, ev: ChatEvent) -> StopEvent:
        memory = await ctx.store.get("memory")
        parts: list[str] = []
        async for chunk in await self.llm.astream_chat(memory.get()):
            delta = chunk.delta or ""
            ctx.write_event_to_stream(StreamEvent(delta=delta))
            parts.append(delta)
        answer = "".join(parts)
        memory.put(ChatMessage(role="assistant", content=answer))
        await ctx.store.set("memory", memory)
        return StopEvent(result={"response": answer, "score": None, "sources": []})

    # ── step 2b: clarify (HUMAN IN THE LOOP) ───────────────────────────────────

    @step
    async def clarify(self, ctx: Context, ev: ClarifyEvent) -> RetrieveEvent:
        # 1) Build a topic-aware clarifying question (topics may be 0, 1, or many).
        opts = [_TOPIC_OPTIONS.get(t, "") for t in ev.topics]
        options = ", ".join(o for o in opts if o) or "—"
        topic_str = ", ".join(ev.topics) if ev.topics else "none"
        question = await self.llm.apredict(
            _CLARIFY_TMPL,
            query=ev.original_query,
            topic=topic_str,
            options=options,
        )
        question = question or _CLARIFY_FALLBACK

        # 2) Pause the run: emit the question to the stream as an
        #    InputRequiredEvent and wait until a HumanResponseEvent arrives.
        response = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_id=ev.original_query,                  # unique id for this wait
            waiter_event=InputRequiredEvent(prefix=question),
        )
        reply = (response.response or "").strip()
        print(f"[clarify] question asked, reply received: {reply!r}")

        # 3) Record both sides of the exchange in memory.
        memory = await ctx.store.get("memory")
        memory.put(ChatMessage(role="assistant", content=question))
        memory.put(ChatMessage(role="user", content=reply))
        await ctx.store.set("memory", memory)

        # 4) Resume straight into retrieval with the merged, now-specific query.
        merged = f"{ev.original_query}\nНэмэлт тодруулга: {reply}"
        await ctx.store.set("original_query", merged)
        return RetrieveEvent(query=merged, topics=ev.topics, retry=0)

    # ── step 3: decompose complex query (fan-out) ──────────────────────────────

    @step
    async def decompose(self, ctx: Context, ev: ComplexEvent) -> SubQueryEvent:
        result: DecomposedQueries = await self.llm.astructured_predict(
            DecomposedQueries, _DECOMPOSE_TMPL,
            query=ev.query, max_subqueries=MAX_SUBQUERIES,
        )
        sub_queries = (result.sub_queries or [ev.query])[:MAX_SUBQUERIES]
        print(f"[decompose] {len(sub_queries)} sub-queries: {sub_queries}")
        await ctx.store.set("expected_results", len(sub_queries))

        for i, sq in enumerate(sub_queries[:-1]):
            ctx.send_event(SubQueryEvent(query=sq, topics=ev.topics, index=i))
        return SubQueryEvent(query=sub_queries[-1], topics=ev.topics, index=len(sub_queries) - 1)

    # ── step 4: retrieve (generic; runs in parallel for sub-queries) ───────────

    @step(num_workers=4)
    async def retrieve(
        self, ctx: Context, ev: RetrieveEvent | SubQueryEvent
    ) -> RetrievalResultEvent:
        if isinstance(ev, RetrieveEvent):
            await ctx.store.set("expected_results", 1)
            index = 0
        else:
            index = ev.index

        # `topics` is forwarded as an opaque filter list; empty list = no filter.
        context = await self.retriever(ev.query, ev.topics)
        return RetrievalResultEvent(context=context or "", index=index)

    # ── step 5: collect + synthesize ───────────────────────────────────────────

    @step
    async def collect_and_synthesize(
        self, ctx: Context, ev: RetrievalResultEvent
    ) -> AnswerEvent | None:
        expected: int = await ctx.store.get("expected_results", default=1)
        results = ctx.collect_events(ev, [RetrievalResultEvent] * expected)
        if results is None:
            return None  # wait for the remaining fan-out branches

        combined = "\n\n---\n\n".join(r.context for r in results if r.context.strip())
        query: str = await ctx.store.get("original_query")

        stream = await self.llm.astream_chat(messages=[
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=_SYNTH_TMPL.format(query=query, context=combined)),
        ])
        parts: list[str] = []
        async for chunk in stream:
            delta = chunk.delta or ""
            ctx.write_event_to_stream(StreamEvent(delta=delta))
            parts.append(delta)

        return AnswerEvent(query=query, context=combined, answer="".join(parts))

    # ── step 6: evaluate (score 1-10) + save to history ────────────────────────

    @step
    async def evaluate(self, ctx: Context, ev: AnswerEvent) -> StopEvent:
        evaluation: AnswerEvaluation = await self.llm.astructured_predict(
            AnswerEvaluation, _EVALUATE_TMPL,
            query=ev.query, context=ev.context, answer=ev.answer,
        )
        score = max(1, min(10, evaluation.score))
        print(f"[evaluate] score={score}/10 verdict={evaluation.verdict} :: {evaluation.reason}")

        answer = ev.answer if evaluation.verdict != "unanswerable" else _NO_INFO

        # Save answer + score into chat history (rides on the assistant message)
        # and append to a persistent score log for offline evaluation.
        memory = await ctx.store.get("memory")
        memory.put(ChatMessage(
            role="assistant",
            content=answer,
            additional_kwargs={"rag_score": score, "verdict": evaluation.verdict},
        ))
        await ctx.store.set("memory", memory)

        score_log: list[dict] = await ctx.store.get("score_history", default=[])
        score_log.append({
            "query": ev.query,
            "score": score,
            "verdict": evaluation.verdict,
            "reason": evaluation.reason,
        })
        await ctx.store.set("score_history", score_log)

        return StopEvent(result={
            "response": answer,
            "score": score,
            "verdict": evaluation.verdict,
            "context": ev.answer,        # retrieved RAG context, for history/debugging
            "sources": [],
        })
