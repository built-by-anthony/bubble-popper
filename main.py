from fred_ingest import ingest_fred
from edgar_ingest import ingest_edgar


def main():
    print("=== FRED ===")
    ingest_fred()

    print("\n=== EDGAR ===")
    ingest_edgar()


if __name__ == "__main__":
    main()
