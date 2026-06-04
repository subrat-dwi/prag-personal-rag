from llm.chat_llm import chat_with_llm
from ingestion.drive_sync import sync_drive
import sys
import logging
from config.settings import settings
import os
from dotenv import load_dotenv

#---------------Configure logging-------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.WARNING),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()  # load .env variables into environment

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "personal-rag")

def main():
    #-----------sync Drive and ingest any changes before chat starts-----------
    print("\nSyncing documents from Google Drive...")
    try:
        sync_drive()
        print("Sync complete.\n")
    except Exception as e:
        logger.error("Drive sync failed: %s", e)
        print(f"Warning: Drive sync failed ({e}). Continuing with existing index.\n")
        # don't exit — existing Qdrant data is still usable

    #--------------start chat---------------------
    print("Personal RAG — ask anything about your documents.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            sys.exit(0)

        if not query:
            continue

        if query.lower() in {"exit", "quit", "bye"}:
            print("Bye.")
            break

        try:
            answer = chat_with_llm(query)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            logger.error("Query failed: %s", e)
            print(f"\nSomething went wrong: {e}\n")

if __name__ == "__main__":
    main()