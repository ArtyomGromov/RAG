import os
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import uuid
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class DenseQdrantRAG:
    def __init__(self, qdrant_client: QdrantClient, model_name="all-MiniLM-L6-v2",
                 use_gpu=False, llm_model="google/flan-t5-small"):
        self.qdrant = qdrant_client
        self.encoder = SentenceTransformer(model_name, device='cuda' if use_gpu else 'cpu')
        
        # Загружаем LLM (Flan-T5-small)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
        self.device = 'cuda' if use_gpu else 'cpu'
        self.llm = self.llm.to(self.device)

    def _ensure_collection(self, collection):
        try:
            self.qdrant.get_collection(collection)
        except Exception:
            dim = self.encoder.get_sentence_embedding_dimension()
            self.qdrant.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def _extract_chunks(self, pdf_path: str, doc_name: str, chunk_size=256, overlap=50):
        laparams = LAParams()
        chunks = []
        for i, layout in enumerate(extract_pages(pdf_path, laparams=laparams)):
            text = ''.join(el.get_text() for el in layout if isinstance(el, LTTextContainer))
            words = text.split()
            start = 0
            while start < len(words):
                chunk_words = words[start:start + chunk_size]
                chunk_text = f"{doc_name}: " + ' '.join(chunk_words)
                if len(chunk_text) > 20:
                    chunks.append({'text': chunk_text, 'page': i + 1, 'document': doc_name})
                start += (chunk_size - overlap)
        return chunks

    # Функция 1: Загрузка PDF в коллекцию
    def add_pdf(self, pdf_path, collection, chunk_size=256, overlap=0):
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self._ensure_collection(collection)
        chunks = self._extract_chunks(pdf_path, doc_name, chunk_size, overlap)
        texts = [c['text'] for c in chunks]
        vectors = self.encoder.encode(texts, convert_to_numpy=True)
        points = [
            {"id": str(uuid.uuid4()), "vector": vec.tolist(),
             "payload": {"text": chunk['text'], "page": chunk['page'], "document": chunk['document']}}
            for vec, chunk in zip(vectors, chunks)
        ]
        self.qdrant.upsert(collection_name=collection, points=points)
        print(f"Загружено {len(points)} чанков из {pdf_path} в коллекцию {collection}")

    # Функция 2
    def generate_answer(self, query, collection, top_k=2, max_tokens=256):
        # 1. dense поиск по коллекции
        
        vec = self.encoder.encode([query], convert_to_numpy=True)[0]
        results = self.qdrant.search(
            collection_name=collection,
            query_vector=vec.tolist(),
            limit=top_k,
            with_payload=True
        )

        # 2. Собираем контекст из найденных чанков (top_k)
        context = "\n\n".join([r.payload['text'] for r in results if 'text' in r.payload])

        # 3. Формируем prompt для LLM (Flan-T5)
        prompt = f"""Answer the following question **only** in English and **only** using the information 
        from the provided context. Do not make up any information. If the 
        answer is not in the context, reply "Not found in the provided context."
        Question: {query}
        Context:
        {context}
        Answer in English:"""

        # 4. Генерируем ответ
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        output = self.llm.generate(**inputs, max_new_tokens=max_tokens)
        answer = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)

        # 5. Возвращаем все нужное для чата/бота
        return {
            "question": query,
            "chunks": [r.payload['text'] for r in results],
            "answer": answer.strip()
        }
