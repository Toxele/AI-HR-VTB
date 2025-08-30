import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from typing import Dict, List, Tuple, Union
from document_managament.document_factory import DocumentFactory
from document_managament.word2vec_processor import Word2VecProcessor
from document_managament.pretrained_embeddings import PretrainedEmbeddings
import re
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentManager:
    def __init__(self,
                 text_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 rebuild_all_indexes=False,
                 use_word2vec=True,
                 use_pretrained=False,  # Новый параметр
                 pretrained_model_type: str = "word2vec"  # Тип предобученной модели
                 ):
        self.text_embedding_model = text_embedding_model
        self.text_embeddings = HuggingFaceEmbeddings(model_name=self.text_embedding_model)

        # Получаем абсолютный путь к корню проекта
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        self.base_documents_path = str(project_root / 'documents')

        self.resume_documents = []
        self.job_requirement_documents = []
        self.rebuild_all_indexes = rebuild_all_indexes
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

        # Настройка эмбеддингов
        self.use_pretrained = use_pretrained
        self.use_word2vec = use_word2vec if not use_pretrained else False
        self.pretrained_model_type = pretrained_model_type

        self.word2vec_processor = None
        self.pretrained_embeddings = None

        self._setup_embeddings()
        self.load_documents()

        if not self.use_pretrained:
            self._train_word2vec_model()

    def _setup_embeddings(self):
        """Настройка системы эмбеддингов"""
        if self.use_pretrained:
            print("Использование предобученных эмбеддингов...")
            self.pretrained_embeddings = PretrainedEmbeddings(self.pretrained_model_type)
            self.use_word2vec = False
        elif self.use_word2vec:
            print("Использование Word2Vec с обучением на данных...")
            self.word2vec_processor = Word2VecProcessor()
        else:
            print("Использование только TF-IDF...")
            self.word2vec_processor = None

    def load_documents(self):
        """Загружает документы из четко разделенных папок"""
        print("Начинаем загрузку документов...")
        print(f"Базовый путь к документам: {self.base_documents_path}")
        print(
            f"Режим эмбеддингов: {'Pretrained' if self.use_pretrained else 'Word2Vec' if self.use_word2vec else 'TF-IDF only'}")

        # Загружаем резюме из папки cv/
        cv_path = Path(self.base_documents_path) / 'cv'
        if cv_path.exists():
            print("\nЗагрузка резюме кандидатов...")
            self.resume_documents = self._load_documents_from_folder(cv_path, "резюме")
        else:
            print(f"Папка с резюме не найдена: {cv_path}")

        # Загружаем вакансии из папки vacancy/
        vacancy_path = Path(self.base_documents_path) / 'vacancy'
        if vacancy_path.exists():
            print("\nЗагрузка описаний вакансий...")
            self.job_requirement_documents = self._load_documents_from_folder(vacancy_path, "вакансия")
        else:
            print(f"Папка с вакансиями не найдена: {vacancy_path}")

        print(f"\nИтог загрузки:")
        print(f"Резюме кандидатов: {len(self.resume_documents)}")
        print(f"Описания вакансий: {len(self.job_requirement_documents)}")

    def _load_documents_from_folder(self, base_folder: Path, doc_type: str) -> List:
        """Загружает документы из указанной папки и всех ее подпапок"""
        documents = []
        supported_extensions = ['.pdf', '.docx', '.rtf']

        # Рекурсивно ищем все файлы с поддерживаемыми расширениями
        for extension in supported_extensions:
            folder_path = base_folder / extension[1:]  # pdf, docx, rtf
            if folder_path.exists():
                print(f"  Сканируем папку: {folder_path}")
                files = list(folder_path.glob(f'*{extension}'))
                print(f"    Найдено {len(files)} файлов {extension}")

                for file_path in files:
                    if file_path.is_file():
                        print(f"    Обрабатываем {doc_type}: {file_path.name}")

                        document = DocumentFactory.create_document(
                            document_name=file_path.name,
                            document_dir=str(file_path.parent),
                            extract_dir_name=file_path.stem,
                            text_embeddings=self.text_embeddings,
                            rebuild_index=self.rebuild_all_indexes
                        )

                        if document:
                            documents.append(document)
                            print(f"      ✓ Успешно загружен")
                        else:
                            print(f"      ✗ Не удалось загрузить")
            else:
                print(f"  Папка {folder_path} не существует")

        return documents

    def _train_word2vec_model(self):
        """Обучение Word2Vec модели на всех документах"""
        if not self.use_word2vec or not self.word2vec_processor:
            return

        print("\nОбучение Word2Vec модели...")

        # Собираем все тексты для обучения
        all_texts = []

        # Тексты вакансий
        for job_doc in self.job_requirement_documents:
            texts = job_doc.get_all_text()
            if texts:
                all_texts.extend(texts)

        # Тексты резюме
        for resume_doc in self.resume_documents:
            texts = resume_doc.get_all_text()
            if texts:
                all_texts.extend(texts)

        if not all_texts:
            print("Недостаточно данных для обучения Word2Vec модели")
            return

        # Добавляем дополнительные IT-термины для улучшения обучения
        it_terms = [
            "Python Java JavaScript TypeScript SQL NoSQL HTML CSS React Angular Vue",
            "Django Flask Spring Hibernate Node.js Express Laravel RubyOnRails",
            "Docker Kubernetes AWS Azure GCP DevOps CI/CD Jenkins GitLab GitHub",
            "Linux Windows macOS Ubuntu CentOS Debian RedHat Apache Nginx",
            "MySQL PostgreSQL MongoDB Redis Elasticsearch Cassandra Oracle",
            "машинное обучение искусственный интеллект нейронные сети数据分析",
            "веб разработка мобильная разработка фронтенд бэкенд фуллстек",
            "базы данных облачные технологии микросервисы API REST GraphQL",
            "тестирование качество обеспечение автоматическое тестирование QA",
            "аггиле скрам канбан проектный менеджмент управление продуктом"
        ]

        all_texts.extend(it_terms)

        print(f"Обучение на {len(all_texts)} текстах...")
        self.word2vec_processor.train_model(all_texts)

        vocab_size = self.word2vec_processor.get_vocabulary_size()
        print(f"Размер словаря Word2Vec: {vocab_size}")

        if vocab_size < 50:
            print("Предупреждение: Слишком маленький словарь. Качество сопоставления может быть низким.")

    def _calculate_semantic_similarity(self, job_text: str, resume_text: str) -> float:
        """Вычисление семантической схожести в зависимости от режима"""
        if self.use_pretrained and self.pretrained_embeddings:
            # Используем предобученные эмбеддинги
            return self.pretrained_embeddings.calculate_similarity(job_text, resume_text)
        elif self.use_word2vec and self.word2vec_processor:
            # Используем Word2Vec
            return self.word2vec_processor.calculate_semantic_similarity(job_text, resume_text)
        else:
            # Используем только TF-IDF
            return self._calculate_tfidf_similarity(job_text, resume_text)

    def _calculate_tfidf_similarity(self, job_text: str, resume_text: str) -> float:
        """Вычисление TF-IDF схожести между двумя текстами"""
        try:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([job_text, resume_text])

            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1]).flatten()[0]
            return float(similarity)
        except Exception as e:
            print(f"Ошибка при вычислении TF-IDF схожести: {e}")
            return 0.0

    def _calculate_hybrid_similarity(self, job_text: str, resume_text: str) -> float:
        """Гибридный расчет схожести: TF-IDF + Semantic"""
        # TF-IDF схожесть
        tfidf_similarity = self._calculate_tfidf_similarity(job_text, resume_text)

        # Семантическая схожесть
        semantic_similarity = self._calculate_semantic_similarity(job_text, resume_text)

        print(f"  TF-IDF: {tfidf_similarity:.3f}, Semantic: {semantic_similarity:.3f}")

        # Комбинируем с весами
        hybrid_similarity = (tfidf_similarity * 0.6) + (semantic_similarity * 0.4)

        return hybrid_similarity

    def match_candidates_to_job(self, top_n: int = 5) -> List[Tuple[float, Dict]]:
        """Сопоставляет кандидатов с требованиями вакансии"""
        print(
            f"\nНачинаем сопоставление: {len(self.job_requirement_documents)} вакансий, {len(self.resume_documents)} резюме")

        if not self.job_requirement_documents:
            print("Нет документов с требованиями вакансий!")
            return []

        if not self.resume_documents:
            print("Нет резюме кандидатов!")
            return []

        # Для каждой вакансии находим лучших кандидатов
        all_results = []

        for job_doc in self.job_requirement_documents:
            job_text = " ".join(job_doc.get_all_text())
            if not job_text:
                continue

            print(f"\nАнализируем вакансию: {job_doc.document_name}")

            # Получаем тексты резюме
            resume_texts = []
            resume_metadata = []

            for resume_doc in self.resume_documents:
                texts = resume_doc.get_all_text()
                if texts:
                    resume_texts.append(" ".join(texts))
                    resume_metadata.append({
                        'document_name': resume_doc.document_name,
                        'source': resume_doc.document_path,
                        'file_type': getattr(resume_doc, 'document_name', '').split('.')[-1]
                    })

            if not resume_texts:
                continue

            # Вычисляем схожесть
            try:
                similarities = []
                for i, resume_text in enumerate(resume_texts):
                    similarity = self._calculate_vacancy_specific_similarity(
                        job_text, resume_text, job_doc.document_name
                    )
                    similarities.append(similarity)
                    print(f"  Кандидат {i + 1}: {similarity:.3f}")

                # Сортируем по убыванию схожести
                job_results = []
                for i, similarity in enumerate(similarities):
                    job_results.append((similarity, {
                        'similarity_score': float(similarity),
                        'resume_text': resume_texts[i],
                        'metadata': resume_metadata[i],
                        'job_text': job_text,
                        'job_name': job_doc.document_name
                    }))

                job_results.sort(key=lambda x: x[0], reverse=True)
                all_results.extend(job_results[:top_n])

            except Exception as e:
                print(f"Ошибка при вычислении схожести: {e}")
                continue

        # Сортируем общие результаты
        all_results.sort(key=lambda x: x[0], reverse=True)
        print(f"Успешно проанализировано {len(all_results)} кандидатов")
        return all_results[:top_n]

    def _calculate_vacancy_specific_similarity(self, job_text: str, resume_text: str, job_title: str) -> float:
        """Вычисление схожести с учетом специфики вакансии"""
        if self.use_pretrained and self.pretrained_embeddings:
            # Используем улучшенную схожесть от pretrained эмбеддингов
            return self.pretrained_embeddings.calculate_vacancy_specific_similarity(
                job_text, resume_text, job_title
            )
        else:
            # Стандартная схожесть
            return self._calculate_hybrid_similarity(job_text, resume_text)

    def get_documents_info(self) -> Dict:
        """Возвращает информацию о загруженных документах и эмбеддингах"""
        info = {
            'resumes_count': len(self.resume_documents),
            'vacancies_count': len(self.job_requirement_documents),
            'embedding_mode': 'pretrained' if self.use_pretrained else 'word2vec' if self.use_word2vec else 'tfidf_only'
        }

        if self.use_pretrained and self.pretrained_embeddings:
            info.update({
                'pretrained_model_loaded': self.pretrained_embeddings.is_model_loaded(),
                'vocabulary_size': self.pretrained_embeddings.get_vocabulary_size()
            })
        elif self.use_word2vec and self.word2vec_processor:
            info.update({
                'word2vec_vocabulary_size': self.word2vec_processor.get_vocabulary_size(),
                'word2vec_usable': self.word2vec_processor.get_vocabulary_size() >= 20
            })

        return info


# Пример использования
if __name__ == "__main__":
    print("Тестирование DocumentManager с предобученными эмбеддингами...")

    # Вариант 1: С предобученными эмбеддингами
    document_manager = DocumentManager(
        rebuild_all_indexes=False,
        use_pretrained=True,  # Используем предобученные
        use_word2vec=False  # Отключаем Word2Vec
    )

    # Вариант 2: С Word2Vec обучением
    # document_manager = DocumentManager(
    #     rebuild_all_indexes=False,
    #     use_pretrained=False,  # Не используем предобученные
    #     use_word2vec=True      # Используем Word2Vec
    # )

    # Информация о загруженных документах
    info = document_manager.get_documents_info()
    print(f"Информация о документах: {info}")

    matches = document_manager.match_candidates_to_job(top_n=3)

    if matches:
        for score, candidate_info in matches:
            print(f"Схожесть: {score:.3f}")
            print(f"Кандидат: {candidate_info['metadata']['document_name']}")
            print(f"Вакансия: {candidate_info['job_name']}")
            print("---")
    else:
        print("Не найдено совпадений")