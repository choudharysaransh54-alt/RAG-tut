from pathlib import Path
from typing import Any

import chromadb


class ChromaVectorStore:
	def __init__(
		self,
		persist_directory: str | Path,
		collection_name: str = "pdf_documents",
	) -> None:
		self.client = chromadb.PersistentClient(path=str(persist_directory))
		try:
			self.collection = self.client.get_collection(name=collection_name)
		except Exception as error:
			raise RuntimeError(
				f"Chroma collection '{collection_name}' was not found in "
				f"'{persist_directory}'."
			) from error

		if self.collection.count() == 0:
			raise RuntimeError(f"Chroma collection '{collection_name}' is empty.")

	def query(self, embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
		result_count = min(top_k * 4, self.collection.count())
		result = self.collection.query(
			query_embeddings=[embedding],
			n_results=result_count,
			include=["documents", "metadatas", "distances"],
		)

		documents = result["documents"][0] or []
		metadatas = result["metadatas"][0] or []
		distances = result["distances"][0] or []
		matches = []
		seen_documents = set()
		for content, metadata, distance in zip(documents, metadatas, distances):
			if not content or content in seen_documents:
				continue
			seen_documents.add(content)
			matches.append({
				"content": content,
				"metadata": metadata or {},
				"score": 1 - distance,
			})
			if len(matches) == top_k:
				break
		return matches
