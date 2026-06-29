from enum import Enum
import json


class Topic(str, Enum):
    """Single source of truth for the retrieval topic taxonomy.

    The string values MUST match the `topic` column in the FAQ dataset exactly —
    retrieval filters by string equality on node metadata. Both the router
    (workflow `RouteDecision`) and the retriever's data-validation check are
    derived from this enum, so the taxonomy lives in exactly one place.

    Inherits from `str`, so a member IS its value: `Topic.TAX == "tax"`, and it
    can be used anywhere a plain string is expected (JSON, metadata, joins)
    without `.value`.

    NONE is a sentinel meaning "no topic filter" (search all). It is NOT a data
    topic; the router signals it with an empty list rather than emitting NONE.
    """

    NEW_COMPANY = "new_company"
    COMPANY_VERIFICATION = "company_verification"
    PROPERTY = "property"
    VEHICLE = "vehicle"
    FINANCIAL_COMPLIANCE = "financial_compliance"
    TAX = "tax"
    INSURANCE = "insurance"
    REGULATORY_PERMITS = "regulatory_permits"
    E_SIGN = "e-sign"
    GENERAL = "general"
    SYSTEM = "system"
    NEW_E_BUSINESS = "new_e-business"
    OTHER = "other"
    NONE = "none"

def append_session_to_history(
    file_path: str, session_id: str, user_id: str, start_timestamp: str, messages: list
):
    """Custom function to append entire session as a dict to JSON file list."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    session_entry = {
        "session_id": session_id,
        "user_id": user_id,
        "start_timestamp": start_timestamp,
        "messages": messages,
    }
    history.append(session_entry)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)