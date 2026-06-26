from fred_ingest import ingest_fred
from edgar_ingest import ingest_edgar
from shiller_ingest import ingest_shiller


def main():
    print("=== FRED ===")
    ingest_fred()

    print("\n=== EDGAR ===")
    ingest_edgar()

    print("\n=== SHILLER ===")
    ingest_shiller()


if __name__ == "__main__":
    main()
