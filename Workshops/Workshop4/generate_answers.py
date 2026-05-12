"""
generate_answers.py — Itera las 50 preguntas y guarda my_rag_output.json.
Requiere haber corrido ingest.py primero.
"""

import json
import time
from rag_pipeline import rag_chain, retriever

INPUT_FILE  = "docs/questions.json"
OUTPUT_FILE = "my_rag_output.json"

def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    # Extraer solo el texto de la pregunta (el JSON tiene objetos con "question")
    if isinstance(questions[0], dict):
        question_texts = [q["question"] for q in questions]
    else:
        question_texts = questions

    print(f"📋 Respondiendo {len(question_texts)} preguntas...\n")

    results = []
    for i, question in enumerate(question_texts, 1):
        print(f"  [{i:02d}/{len(question_texts)}] {question[:70]}...")

        try:
            source_docs = retriever.invoke(question)
            answer      = rag_chain.invoke(question)
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            source_docs = []
            answer      = f"Error: {e}"

        results.append({
            "question": question,
            "answer":   answer,
            "contexts": [doc.page_content for doc in source_docs],
        })

        # Pequeña pausa para no saturar el rate limit de NIM
        time.sleep(0.5)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Resultados guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
