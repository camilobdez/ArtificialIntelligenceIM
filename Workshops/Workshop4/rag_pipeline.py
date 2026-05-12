"""
rag_pipeline.py — Fase B: carga el índice Qdrant y construye el pipeline RAG con NVIDIA NIM.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.llms import LLM
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION   = "karpathy_qdrant"
PERSIST_PATH = "./db/karpathy_qdrant"
MODEL        = "meta/llama-3.3-70b-instruct"

# ── 1. LLM: NVIDIA NIM via cliente OpenAI-compatible ──────────────────────
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

class NvidiaLLM(LLM):
    model: str = MODEL

    @property
    def _llm_type(self) -> str:
        return "nvidia-nim"

    def _call(
        self,
        prompt: str,
        stop=None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs,
    ) -> str:
        response = nvidia_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content

llm = NvidiaLLM()
print(f"✅ LLM: NVIDIA NIM / {MODEL}")

# ── 2. Embeddings — mismo modelo que en la ingesta ────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ── 3. Cargar Qdrant desde disco ──────────────────────────────────────────
qdrant_client = QdrantClient(path=PERSIST_PATH)
vectorstore   = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION,
    embedding=embeddings,
)
print("✅ Vector store cargado desde disco")

# ── 4. Retriever ──────────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.3},
)

# ── 5. Pipeline LCEL ──────────────────────────────────────────────────────
def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Fragmento {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )

prompt = ChatPromptTemplate.from_template("""
You are an expert assistant. Answer the question based ONLY on the provided context.
If the information is not in the context, explicitly say you don't have it.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
""")

rag_chain = (
    {
        "context":  retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
