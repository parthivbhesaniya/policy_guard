import os

# Unit tests must never talk to LangSmith. A developer's .env usually turns tracing on, and some
# modules under test call load_dotenv() on import -- which doesn't override a variable that's
# already set, so switching tracing off here, before any test module is imported, wins.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
