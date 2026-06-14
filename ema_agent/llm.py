from llama_index.llms.openai import OpenAI
# from llama_index.llms.gemini import Gemini
# from llama_index.llms.ollama import Ollama
import os
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    # max_tokens=4000,
    max_retries=2,
    timeout=20,
    api_key=OPENAI_API_KEY,
)

llm3 = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    # max_tokens=4000,
    max_retries=2,
    timeout=20,
    api_key=OPENAI_API_KEY,
)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# llm2 = Gemini(
#     model="models/gemini-1.5-pro",
#     temperature=0.1,
#     max_tokens=4000,
#     max_retries=2,
#     timeout=20,
#     api_key=GOOGLE_API_KEY,  # uses GOOGLE_API_KEY env var by default
# )

# from llama_index.llms.google_genai import GoogleGenAI
# from google.genai import types

# llm = GoogleGenAI(
#     # model="gemini-2.5-flash",
#     model="gemini-1.5-pro",
#     generation_config=types.GenerateContentConfig(
#         thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
#     ),
#     temperature=0.1,
#     max_tokens=4000,
#     max_retries=2,
#     timeout=20,
#     api_key=GOOGLE_API_KEY,  # uses GOOGLE_API_KEY env var by default
# )


# llm3 = Ollama(
#     # model="gemma3:12b",
#     # model="llama3.1:70b",
#     model="gpt-oss:20b",
#     temperature=0.2,
#     request_timeout=120.0,
#     # base_url="http://202.70.34.10:11434",
#     base_url="http://localhost:11434",
#     # Manually set the context window to limit memory usage
#     context_window=8000,
# )


async def main():
    response = llm.complete("Hello, world!")
    print(response.raw)


# if __name__ == "__main__":
#     import asyncio

#     asyncio.run(main())
