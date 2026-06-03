from ingestion.parsers import parse_file
from ingestion.pipeline import ingest_file
from llm.chat_llm import chat_with_llm

def main():
    # print(parse_file(input("Enter file path: ")))
    # ingest_file(input("Enter file path: "))

    while True:
        query = input("Ask for information: ")
        if query.lower() == "bye":
            break
        response = chat_with_llm(query)
        print("PRAG: ", response)

if __name__ == "__main__":
    main()