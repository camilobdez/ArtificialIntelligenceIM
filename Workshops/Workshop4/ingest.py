"""
ingest.py — Fase A: carga la transcripción, genera chunks y guarda el índice Qdrant.
Ejecutar una sola vez antes de rag_pipeline.py.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os

COLLECTION   = "karpathy_qdrant"
PERSIST_PATH = "./db/karpathy_qdrant"
DOC_PATH     = "./docs/intro-to-llms-karpathy.txt"

def main():
    # 1. Cargar y dividir el documento
    loader    = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = splitter.split_documents(documents)
    print(f"✅ {len(docs)} chunks generados")

    # 2. Embeddings locales — all-mpnet-base-v2 produce vectores de dim 768
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # 3. Crear colección Qdrant persistente en disco
    os.makedirs(PERSIST_PATH, exist_ok=True)
    qdrant_client = QdrantClient(path=PERSIST_PATH)
    qdrant_client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )
    vectorstore.add_documents(docs)
    print(f"✅ Base de datos vectorial creada en {PERSIST_PATH}")

if __name__ == "__main__":
    main()
