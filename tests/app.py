from llama_index.core.workflow import (
    Context,
    HumanResponseEvent,
    InputRequiredEvent,
)
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import sys

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))
from ema_agent.llm import llm
from ema_agent.workflow import RouterWorkflow, StreamEvent
from ema_agent.store_index_faq import retrieve_documents as faq_retrieve

    
SYSTEM_PROMPT = """
You are a government service support assistant - e-business application.

Your job:
- Understand the user's intent
- Decide whether you need to call a tool
- Use tools ONLY when necessary
- Never hallucinate answers

Rules:
- If the user asks for factual information, retrieve it from knowledge tools
- If multiple services or documents match, ask the user to choose
- If no relevant data is found, say you do not have that information
- Always answer in Mongolian
"""

# "Answer ONLY using the retrieved context below. Do not add anything not present in the context."
# "If the retrieved context does not directly answer the question, say: Энэ асуултанд хариулах мэдээлэл байхгүй байна."

# Replace this stub with your real hybrid + rerank retriever.
# `topics` is now a list (0, 1, or many). faq_retrieve takes a single topic,
# so bridge by passing the first topic (or None when the list is empty).
async def example_retriever(query: str, topics: list[str]) -> str:
    # print(f"[example_retriever] got query: {query!r} topics: {topics}")
    topic = topics[0] if topics else None
    context = await faq_retrieve(query, topic, use_reranker=True)
    # print(f"[example_retriever] got context: {context!r}")
    return context

async def main():
    # agent = RouterWorkflow(
    #     llm=llm,
    #     system_prompt=SYSTEM_PROMPT,
    #     timeout=120,
    #     verbose=True,
    # )
    # ctx = Context(agent)

    # timeout must cover human think-time, since the HITL run stays alive while
    # waiting on wait_for_event. Bump it up for interactive use.
    agent = RouterWorkflow(
        llm=llm,
        system_prompt=SYSTEM_PROMPT, 
        retriever=example_retriever, 
        timeout=600, 
        verbose=True)
    ctx = Context(agent)  # reused across turns so memory + score log persist

    while True:
        text_input = input("User: ")
        if text_input.strip() == "exit":
            break

        handler = agent.run(ctx=ctx, input=text_input)
        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                # The workflow paused inside clarify and wants a human reply.
                reply = input("\n[clarify] " + event.prefix + "\nUser: ")
                handler.ctx.send_event(HumanResponseEvent(response=reply))
            elif isinstance(event, StreamEvent):
                print(event.delta, end="", flush=True)

        result = await handler
        if result.get("score") is not None:
            print(f"\n[rag score: {result['score']}/10]")
        print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())