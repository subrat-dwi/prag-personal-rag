from ingestion.parsers import parse_file

def main():
    print(parse_file(input("Enter file path: ")))

if __name__ == "__main__":
    main()