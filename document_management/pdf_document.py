from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
import os
from langchain_community.vectorstores import FAISS
import torch
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import re

try:
    from langchain_community.document_loaders import PyMuPDFLoader

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("PyMuPDFLoader недоступен, будет использована альтернативная обработка PDF")

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("pdfplumber недоступен, будет использована альтернативная обработка PDF")

try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("PyPDF2 недоступен, будет использована альтернативная обработка PDF")


class Pdfdocument:
    def __init__(
            self,
            document_name: str,
            document_dir: str,
            extract_dir_name: str,
            text_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            text_embeddings=None,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            text_storage_name: str = "extracted_text",
            rebuild_index=False
    ):
        self.text_embedding_model = text_embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_name = document_name

        # Очищаем имена директорий от кириллицы и специальных символов
        safe_extract_dir_name = self._make_path_safe(extract_dir_name)
        safe_text_storage_name = self._make_path_safe(text_storage_name)

        # Директории для документа
        extract_dir = os.path.join(document_dir, 'index_' + safe_extract_dir_name)
        self.index_storage_path = os.path.join(extract_dir, safe_text_storage_name)
        self.document_path = os.path.join(document_dir, document_name)

        self.table_lookup = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = None
        self.text_chunks = []

        if text_embeddings is None:
            self.text_embeddings = HuggingFaceEmbeddings(model_name=self.text_embedding_model)
        else:
            self.text_embeddings = text_embeddings

        if Path(self.index_storage_path).is_dir() and not rebuild_index:
            print(f"Загрузка индекса для PDF: {document_name}")
            self._load_index()
        else:
            print(f"Создание индекса для PDF: {document_name}")
            Path(self.index_storage_path).mkdir(parents=True, exist_ok=True)
            self._build_index()

    def _make_path_safe(self, path_string: str) -> str:
        """Преобразует строку в безопасный для путей формат"""
        # Заменяем кириллические символы и специальные символы
        safe_string = re.sub(r'[^a-zA-Z0-9_\-]', '_', path_string)
        # Убираем повторяющиеся подчеркивания
        safe_string = re.sub(r'_+', '_', safe_string)
        # Убираем подчеркивания в начале и конце
        safe_string = safe_string.strip('_')
        # Если строка пустая, используем дефолтное значение
        if not safe_string:
            safe_string = "default_index"
        return safe_string

    def get_all_text(self) -> List[str]:
        """Возвращает весь текст документа в виде списка строк"""
        return [chunk.page_content for chunk in self.text_chunks]

    def search_with_tfidf(self, query: str, top_k: int = 5) -> List[Document]:
        """Поиск с использованием TF-IDF"""
        if self.tfidf_matrix is None:
            texts = self.get_all_text()
            if not texts:
                return []
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        results = []
        for i, similarity in enumerate(similarities):
            if similarity > 0:
                results.append((similarity, self.text_chunks[i]))

        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:top_k]]

    def _load_index(self):
        """Загружает FAISS-индекс и текстовые чанки"""
        try:
            self.db = FAISS.load_local(
                folder_path=self.index_storage_path,
                embeddings=self.text_embeddings,
                allow_dangerous_deserialization=True
            )

            chunks_path = os.path.join(self.index_storage_path, "text_chunks.pkl")
            if os.path.exists(chunks_path):
                with open(chunks_path, "rb") as f:
                    self.text_chunks = pickle.load(f)

        except Exception as e:
            print(f"Ошибка загрузки индекса PDF: {e}")
            self._build_index()

    def _build_index(self):
        """Создает FAISS-индекс и сохраняет текстовые чанки"""
        print(f"Извлечение текста из PDF: {self.document_name}")
        self.text_chunks = self._extract_text()

        if self.text_chunks:
            try:
                self.db = FAISS.from_documents(self.text_chunks, self.text_embeddings)
                self.db.save_local(self.index_storage_path)

                chunks_path = os.path.join(self.index_storage_path, "text_chunks.pkl")
                with open(chunks_path, "wb") as f:
                    pickle.dump(self.text_chunks, f)

                print(f"Индекс PDF создан: {self.document_name}")
            except Exception as e:
                print(f"Ошибка создания индекса FAISS: {e}")
                # Создаем пустой индекс в случае ошибки
                self.db = None
        else:
            print(f"Не удалось извлечь текст из PDF: {self.document_name}")
            self.db = None

    def _extract_text_pymupdf(self):
        """Извлекает текст из PDF с использованием PyMuPDF"""
        try:
            loader = PyMuPDFLoader(self.document_path)
            pages = loader.load()
            return pages
        except Exception as e:
            print(f"Ошибка PyMuPDF: {e}")
            return None

    def _extract_text_pdfplumber(self):
        """Извлекает текст из PDF с использованием pdfplumber"""
        try:
            pages = []
            with pdfplumber.open(self.document_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages.append(Document(
                            page_content=text,
                            metadata={
                                "page": i + 1,
                                "source": self.document_path,
                                "type": "text",
                                "file_type": "pdf"
                            }
                        ))
            return pages
        except Exception as e:
            print(f"Ошибка pdfplumber: {e}")
            return None

    def _extract_text_pypdf2(self):
        """Извлекает текст из PDF с использованием PyPDF2"""
        try:
            pages = []
            with open(self.document_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        pages.append(Document(
                            page_content=text,
                            metadata={
                                "page": i + 1,
                                "source": self.document_path,
                                "type": "text",
                                "file_type": "pdf"
                            }
                        ))
            return pages
        except Exception as e:
            print(f"Ошибка PyPDF2: {e}")
            return None

    def _extract_text(self):
        """Извлекает текст из PDF файла с использованием доступных методов"""
        # Проверяем существование файла
        if not os.path.exists(self.document_path):
            print(f"Файл не найден: {self.document_path}")
            return []

        # Пробуем разные методы извлечения текста
        pages = None

        if PYPDF_AVAILABLE:
            pages = self._extract_text_pymupdf()

        if pages is None and PDFPLUMBER_AVAILABLE:
            pages = self._extract_text_pdfplumber()

        if pages is None and PYPDF2_AVAILABLE:
            pages = self._extract_text_pypdf2()

        if pages is None:
            print("Все методы извлечения текста недоступны. Установите один из: pymupdf, pdfplumber, PyPDF2")
            return []

        # Проверяем, что документ не пустой
        if not pages or not any(page.page_content.strip() for page in pages):
            print(f"PDF документ пуст или не содержит текста: {self.document_path}")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(pages)
        for doc in chunks:
            doc.metadata["type"] = "text"
            doc.metadata["source"] = self.document_path
            doc.metadata["file_type"] = "pdf"
        return chunks

    def search_document(self, query, rerank_k=10):
        """Поиск в документе"""
        if not hasattr(self, 'db') or self.db is None:
            return {"context_docs": [], "omitted_tables": []}

        embedding_results = self.db.similarity_search(query, k=rerank_k)
        tfidf_results = self.search_with_tfidf(query, top_k=rerank_k)

        all_results = embedding_results + tfidf_results

        unique_results = []
        seen_content = set()

        for doc in all_results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_results.append(doc)

        return {
            "context_docs": unique_results[:rerank_k],
            "omitted_tables": []
        }

    def __del__(self):
        if hasattr(self, 'db'):
            del self.db
        torch.cuda.empty_cache()


if __name__ == "__main__":
    document = Pdfdocument(
        document_name='example.pdf',
        document_dir='../documents/cv/pdf/',
        extract_dir_name='example_pdf',
        rebuild_index=False
    )