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


# ── Topic sub-areas (drives the clarifying question) ──────────────────────────

_TOPIC_OPTIONS: dict[str, str] = {
    "new_company": "ХХК/хоршоо/нөхөрлөл байгуулах, нэр авах, бүртгэл, захирал солих, гэрчилгээ, тамга",
    "company_verification": "компанийн бүртгэлийн лавлагаа, хаяг, үүсгэн байгуулагч, эцсийн өмчлөгч, цагдаагийн тодорхойлолт",
    "property": "үл хөдлөх хөрөнгийн лавлагаа, өмчлөл, шилжүүлэг",
    "vehicle": "тээврийн хэрэгслийн лавлагаа, торгууль, хяналтын үзлэг",
    "financial_compliance": "зээлийн мэдээллийн лавлагаа, хугацаа хэтэрсэн зээл, шүүхийн төлбөр",
    "tax": "бүртгэл, тайлан, НӨАТ, төлбөр, татварын төрөл",
    "insurance": "нийгмийн даатгалын бүртгэл, шимтгэл, тодорхойлолт",
    "regulatory_permits": "байгаль орчны нөлөөллийн үнэлгээ, дүгнэлт",
    "e-sign": "тоон гарын үсэг авах, USB токен, PIN/нууц үг, нэвтрэх алдаа",
    "general": "тусгай зөвшөөрөл, тендер, татан буулгах, эрүүл ахуй/гал/ус, үйл ажиллагаа нэмэх",
    "system": "e-business.mn / e-mongolia / ХУР / ДАН систем, бүртгэл, нэвтрэх, төлбөр, баримт",
    "new_e-business": "и-бизнес 2.0, ААН хооронд тээврийн хэрэгсэл шилжүүлэх",
    "other": "",
}


# ── Prompt templates ──────────────────────────────────────────────────────────

_ROUTE_TMPL = PromptTemplate(
    "Classify the user query into a single intent.\n\n"
    "Recent conversation (may be empty):\n{history}\n\n"
    "Query: {query}\n\n"
    "intent:\n"
    "- chat: greeting, thanks, or off-topic smalltalk\n"
    "- clarify: the message only NAMES a topic or area without saying what the user\n"
    "  wants to know — no question, no specific aspect. Nothing specific to answer\n"
    "  yet. Default here for bare topic words.\n"
    "- retrieve: a SPECIFIC answerable question — asks how/what/where/when/how much,\n"
    "  or names a concrete form, deadline, step, or sub-topic.\n"
    "- complex: several distinct sub-questions packed into one message.\n\n"
    "If the recent conversation already established context, a short follow-up may be\n"
    "a valid 'retrieve' rather than 'clarify'.\n\n"
    "Examples:\n"
    "  'татвар'                  -> clarify  (topic only, no question)\n"
    "  'татвараа яаж төлөх вэ?'   -> retrieve\n"
    "  'компани'                 -> clarify\n"
    "  'ХХК хэрхэн үүсгэх вэ?'    -> retrieve\n"
    "  'НӨАТ-ын тайлан хэзээ?'    -> retrieve\n"
    "  'баярлалаа'               -> chat\n\n"
    "topics (list ALL topics the query clearly concerns — usually one, sometimes\n"
    "two for cross-cutting questions; empty list if none clearly apply. Classify\n"
    "even when intent is 'clarify'). Only include topics genuinely present:\n"
    "- new_company: компани/ХХК/хоршоо/нөхөрлөл үүсгэх, нэр авах, бүртгэл, "
    "захирал/хувьцаа эзэмшигч солих, гэрчилгээ, тамга захиалах\n"
    "- company_verification: хуулийн этгээдийн бүртгэлтэй/бүртгэлгүй лавлагаа, "
    "хаяг, үүсгэн байгуулагч, эцсийн өмчлөгч, цагдаагийн тодорхойлолт\n"
    "- property: үл хөдлөх хөрөнгийн лавлагаа, өмчлөл, шилжүүлэг\n"
    "- vehicle: тээврийн хэрэгслийн лавлагаа, торгууль, хяналтын үзлэг\n"
    "- financial_compliance: зээлийн мэдээллийн лавлагаа, хугацаа хэтэрсэн зээл, "
    "шүүхийн шийдвэрийн төлбөр\n"
    "- tax: татвар, тайлан, НӨАТ, declarations, payments\n"
    "- insurance: нийгмийн даатгал, эрүүл мэндийн даатгал, шимтгэл\n"
    "- regulatory_permits: байгаль орчны нөлөөллийн үнэлгээ/дүгнэлт\n"
    "- e-sign: тоон гарын үсэг, USB token, PIN, esign, нэвтрэх алдаа\n"
    "- general: тусгай зөвшөөрөл, тендер, татан буулгах, эрүүл ахуй/гал/ус, "
    "үйл ажиллагаа нэмэх зэрэг бусад үйлчилгээ\n"
    "- system: e-business.mn / e-mongolia / ХУР / ДАН систем, бүртгэл, нэвтрэх, "
    "төлбөр, баримт байршуулах, техникийн асуудал\n"
    "- new_e-business: и-бизнес (e-business) 2.0, ААН хооронд тээврийн хэрэгсэл "
    "шилжүүлэх\n"
    "- other: in-domain боловч дээрх ангилалд багтахгүй\n"
)

_CLARIFY_TMPL = PromptTemplate(
    "The user sent an unclear message to an e-Mongolia support assistant.\n\n"
    "Message: {query}\n"
    "Detected topic(s): {topic}\n"
    "Known sub-areas for these topic(s) (may be empty): {options}\n\n"
    "Write ONE short clarifying question in Mongolian to find out exactly what they need.\n"
    "- If sub-areas are given, ask which of them they need and list them as options.\n"
    "- If only a topic is hinted with no sub-areas, ask a specific follow-up about it.\n"
    "- If the message is unrelated to government services, politely redirect.\n"
    "- Keep it to 1-2 friendly sentences."
)

_DECOMPOSE_TMPL = PromptTemplate(
    "Break the following complex question into at most {max_subqueries} "
    "focused, self-contained sub-questions.\n\nQuestion: {query}"
)

_SYNTH_TMPL = (
    "Answer the user's question using ONLY the information in the context below.\n"
    "Do not add knowledge outside the context.\n"
    "If the context is empty or does not address the question, say briefly in "
    "Mongolian that you could not find the information.\n"
    "Always respond in Mongolian.\n\n"
    "User question: {query}\n\nContext:\n{context}"
)

_EVALUATE_TMPL = PromptTemplate(
    "Score how well the answer addresses the question using the retrieved context.\n\n"
    "Question: {query}\n\nRetrieved context:\n{context}\n\nAnswer: {answer}\n\n"
    "score: integer 1-10 (10 = fully grounded, complete, directly answers; "
    "1 = wrong, ungrounded, or empty)\n"
    "verdict:\n"
    "- good: complete and grounded in the context\n"
    "- weak_retrieval: answer says no info found but the question looks answerable\n"
    "- unanswerable: genuinely not present in the retrieved context\n"
    "reason: one short sentence."
)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a government service support assistant for e-Mongolia. "
    "Always answer in Mongolian. Never hallucinate."
)

_CLARIFY_FALLBACK = "Уучлаарай, асуултаа арай тодорхой бичиж өгнө үү?"
_NO_INFO = "Уучлаарай, энэ асуултанд хариулах мэдээлэл олдсонгүй."


# ── Workflow ──────────────────────────────────────────────────────────────────

class RouterWorkflow(Workflow):
    def __init__(
        self,
        *args: Any,
        retriever: RetrieverFn,
        llm: FunctionCallingLLM | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.retriever = retriever
        self.llm = llm or OpenAI(model="gpt-4o-mini", temperature=0.1)
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

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
        decision: RouteDecision = await self.llm.astructured_predict(
            RouteDecision, _ROUTE_TMPL, query=query, history=history
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
            "sources": [],
        })
