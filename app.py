from pathlib import Path

from dotenv import load_dotenv

from rag_00.search import RAGSearch

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    try:
        rag_search = RAGSearch(project_root / "data" / "vector_store")
        query = input("Ask a question about your documents: ").strip()
        if query:
            print("\n" + rag_search.search_and_summarize(query, top_k=3))
        else:
            print("No question entered.")
    except Exception as error:
        raise SystemExit(f"Unable to run the RAG app: {error}") from error