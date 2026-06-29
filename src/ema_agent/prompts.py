"""Prompt templates and canned strings for the router workflow.

Extracted from ``workflow.py`` so the control flow there stays readable and so a
consuming app can override the prompts (import this module and reassign, or pass
``system_prompt`` to ``RouterWorkflow``). The topic descriptions here mirror the
``Topic`` taxonomy in ``agent_starter.py``; keep them in sync.
"""

from llama_index.core import PromptTemplate

# ── Topic sub-areas (drives the clarifying question) ──────────────────────────

TOPIC_OPTIONS: dict[str, str] = {
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

ROUTE_TMPL = PromptTemplate(
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

CLARIFY_TMPL = PromptTemplate(
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

DECOMPOSE_TMPL = PromptTemplate(
    "Break the following complex question into at most {max_subqueries} "
    "focused, self-contained sub-questions.\n\nQuestion: {query}"
)

SYNTH_TMPL = (
    "Answer the user's question using ONLY the information in the context below.\n"
    "Do not add knowledge outside the context.\n"
    "If the context is empty or does not address the question, say briefly in "
    "Mongolian that you could not find the information.\n"
    "Always respond in Mongolian.\n\n"
    "User question: {query}\n\nContext:\n{context}"
)

EVALUATE_TMPL = PromptTemplate(
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

DEFAULT_SYSTEM_PROMPT = (
    "You are a government service support assistant for e-Mongolia. "
    "Always answer in Mongolian. Never hallucinate."
)

CLARIFY_FALLBACK = "Уучлаарай, асуултаа арай тодорхой бичиж өгнө үү?"
NO_INFO = "Уучлаарай, энэ асуултанд хариулах мэдээлэл олдсонгүй."
