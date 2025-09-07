import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.corpus.reader import documents
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from typing import Dict, List, Tuple, Union, Any, Optional
from document_management.document_factory import DocumentFactory
from document_management.word2vec_processor import Word2VecProcessor
from document_management.pretrained_embeddings import PretrainedEmbeddings
from document_management.dynamic_scaling_system import DynamicScalingSystem
import re
import logging
from collections import Counter

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentManager:
    def __init__(self,
                 text_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 rebuild_all_indexes=False,
                 use_word2vec=True,
                 use_pretrained=False,
                 pretrained_model_type: str = "word2vec",
                 enable_dynamic_scaling: bool = True,
                 min_domain_frequency: int = 3,
                 min_subdomain_frequency: int = 2):

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

        # Динамическое масштабирование
        self.enable_dynamic_scaling = enable_dynamic_scaling
        self.min_domain_frequency = min_domain_frequency
        self.min_subdomain_frequency = min_subdomain_frequency
        self.dynamic_scaling_system = None

        self.word2vec_processor = None
        self.pretrained_embeddings = None

        self._setup_embeddings()
        self.load_documents()

        if not self.use_pretrained:
            self._train_word2vec_model()

        # Инициализация системы динамического масштабирования
        if self.enable_dynamic_scaling:
            self._initialize_dynamic_scaling()

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

    def _initialize_dynamic_scaling(self):
        """Инициализация системы динамического масштабирования"""
        print("Инициализация системы динамического масштабирования...")

        # Собираем все тексты для анализа доменов
        all_texts = []
        document_types = []

        # Тексты вакансий
        for job_doc in self.job_requirement_documents:
            texts = job_doc.get_all_text()
            if texts:
                all_texts.extend(texts)
                document_types.extend(['vacancy'] * len(texts))

        # Тексты резюме
        for resume_doc in self.resume_documents:
            texts = resume_doc.get_all_text()
            if texts:
                all_texts.extend(texts)
                document_types.extend(['resume'] * len(texts))

        if all_texts:
            self.dynamic_scaling_system = DynamicScalingSystem(
                min_domain_frequency=self.min_domain_frequency,
                min_subdomain_frequency=self.min_subdomain_frequency
            )

            # Анализируем тексты для извлечения доменов
            self.dynamic_scaling_system.analyze_documents(all_texts, document_types)

            # Обновляем веса в pretrained embeddings если они используются
            if self.use_pretrained and self.pretrained_embeddings:
                domain_weights = self.dynamic_scaling_system.get_domain_weights()

                # Вместо прямого вызова update_domain_weights, используем наш публичный метод
                self.update_dynamic_weights(all_texts, document_types)

                print(f"Динамически извлечено доменов: {len(self.dynamic_scaling_system.get_domains())}")
                print(f"Динамически извлечено поддоменов: {len(self.dynamic_scaling_system.get_subdomains())}")

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
            'embedding_mode': 'pretrained' if self.use_pretrained else 'word2vec' if self.use_word2vec else 'tfidf_only',
            'dynamic_scaling_enabled': self.enable_dynamic_scaling
        }

        if self.enable_dynamic_scaling and self.dynamic_scaling_system:
            info.update({
                'domains_count': len(self.dynamic_scaling_system.get_domains()),
                'subdomains_count': len(self.dynamic_scaling_system.get_subdomains())
            })

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

    def update_dynamic_weights(self, new_texts: List[str], document_types: List[str] = None):
        """Обновление весов на основе новых данных"""
        if not self.enable_dynamic_scaling or not self.dynamic_scaling_system:
            return

        if document_types is None:
            document_types = ['unknown'] * len(new_texts)

        self.dynamic_scaling_system.analyze_documents(new_texts, document_types)

        # Обновляем веса в pretrained embeddings
        if self.use_pretrained and self.pretrained_embeddings:
            domain_weights = self.dynamic_scaling_system.get_domain_weights()
            self.pretrained_embeddings.update_domain_weights(domain_weights)

        print("Динамические веса успешно обновлены")

    def get_domain_analysis(self) -> Dict:
        """Возвращает анализ доменов и поддоменов"""
        if not self.enable_dynamic_scaling or not self.dynamic_scaling_system:
            return {}

        return {
            'domains': self.dynamic_scaling_system.get_domains(),
            'subdomains': self.dynamic_scaling_system.get_subdomains(),
            'domain_weights': self.dynamic_scaling_system.get_domain_weights()
        }

    def get_specific_match(self, resume_name: str, vacancy_name: str) -> Optional[Dict[str, Any]]:
        """
        Находит соответствие конкретного резюме конкретной вакансии

        Args:
            resume_name: Имя файла резюме
            vacancy_name: Имя файла вакансии

        Returns:
            Словарь с информацией о соответствии или None если не найдено
        """
        print(f"\nПоиск соответствия: резюме '{resume_name}' -> вакансия '{vacancy_name}'")

        # Находим резюме
        resume_doc = None
        for doc in self.resume_documents:
            if doc.document_name == resume_name or doc.document_name.startswith(resume_name):
                resume_doc = doc
                break

        if not resume_doc:
            print(f"Резюме '{resume_name}' не найдено")
            return None

        # Находим вакансию
        vacancy_doc = None
        for doc in self.job_requirement_documents:
            if doc.document_name == vacancy_name or doc.document_name.startswith(vacancy_name):
                vacancy_doc = doc
                break

        if not vacancy_doc:
            print(f"Вакансия '{vacancy_name}' не найдена")
            return None

        # Получаем тексты
        resume_text = " ".join(resume_doc.get_all_text())
        vacancy_text = " ".join(vacancy_doc.get_all_text())

        if not resume_text or not vacancy_text:
            print("Один из документов пуст")
            return None

        # Вычисляем схожесть
        try:
            if self.use_pretrained and self.pretrained_embeddings:
                similarity = self.pretrained_embeddings.calculate_vacancy_specific_similarity(
                    vacancy_text, resume_text, vacancy_doc.document_name
                )
            else:
                similarity = self._calculate_hybrid_similarity(vacancy_text, resume_text)

            # Дополнительный анализ
            domain_analysis = self._analyze_domain_compatibility(vacancy_text, resume_text)

            result = {
                'similarity_score': float(similarity),
                'resume_name': resume_doc.document_name,
                'vacancy_name': vacancy_doc.document_name,
                'resume_text_preview': resume_text[:200] + "..." if len(resume_text) > 200 else resume_text,
                'vacancy_text_preview': vacancy_text[:200] + "..." if len(vacancy_text) > 200 else vacancy_text,
                'domain_analysis': domain_analysis,
                'match_level': self._get_match_level(similarity),
                'recommendation': self._get_recommendation(similarity, domain_analysis)
            }

            print(f"Схожесть: {similarity:.3f} ({result['match_level']})")
            return result

        except Exception as e:
            print(f"Ошибка при вычислении схожести: {e}")
            return None

    def _analyze_domain_compatibility(self, vacancy_text: str, resume_text: str) -> Dict[str, Any]:
        """Анализ доменной совместимости"""
        analysis = {
            'shared_keywords': [],
            'missing_keywords': [],
            'domain_compatibility': 0.0
        }

        if self.enable_dynamic_scaling and self.dynamic_scaling_system:
            # Анализ с использованием динамической системы
            vacancy_domain = self.dynamic_scaling_system.suggest_domain_for_text(vacancy_text)
            resume_domain = self.dynamic_scaling_system.suggest_domain_for_text(resume_text)

            analysis['vacancy_domain'] = vacancy_domain
            analysis['resume_domain'] = resume_domain
            analysis['domain_match'] = vacancy_domain == resume_domain

            # Анализ ключевых слов
            vacancy_words = set(self._preprocess_text(vacancy_text))
            resume_words = set(self._preprocess_text(resume_text))

            domain_weights = self.dynamic_scaling_system.get_domain_weights()
            important_keywords = {k: v for k, v in domain_weights.items() if v > 1.5}

            # Общие ключевые слова
            shared = vacancy_words & resume_words & set(important_keywords.keys())
            analysis['shared_keywords'] = list(shared)

            # Отсутствующие ключевые слова
            missing = (vacancy_words - resume_words) & set(important_keywords.keys())
            analysis['missing_keywords'] = list(missing)

            # Совместимость доменов
            if shared:
                analysis['domain_compatibility'] = len(shared) / len(important_keywords)

        return analysis

    def _get_match_level(self, similarity: float) -> str:
        """Определение уровня соответствия"""
        if similarity >= 0.8:
            return "Отличное соответствие"
        elif similarity >= 0.6:
            return "Хорошее соответствие"
        elif similarity >= 0.4:
            return "Удовлетворительное соответствие"
        else:
            return "Низкое соответствие"

    def _get_recommendation(self, similarity: float, domain_analysis: Dict) -> str:
        """Генерация рекомендации"""
        if similarity >= 0.7:
            return "Рекомендуем к рассмотрению. Высокое соответствие требованиям."

        recommendations = []

        if similarity < 0.4:
            recommendations.append("Рассмотреть других кандидатов или пересмотреть требования.")

        if domain_analysis.get('missing_keywords'):
            missing = ", ".join(domain_analysis['missing_keywords'][:3])
            recommendations.append(f"Кандидату не хватает ключевых навыков: {missing}")

        if not domain_analysis.get('domain_match', True):
            recommendations.append("Разные домены специализации. Возможно не оптимальное соответствие.")

        return " ".join(recommendations) if recommendations else "Требуется дополнительная оценка."

    def _preprocess_text(self, text: str) -> List[str]:
        """Предобработка текста для анализа"""
        if not text:
            return []

        text = text.lower()
        text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
        words = text.split()

        # Базовые стоп-слова
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до'}
        words = [word for word in words if word not in stop_words and len(word) > 2]

        return words

    def load_document(self, file_path: str):
        """
        Загружает один документ по указанному пути

        Args:
            file_path: Полный путь к файлу

        Returns:
            Загруженный документ или None если не удалось загрузить
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                print(f"Файл не найден: {file_path}")
                return None

            document_name = file_path_obj.name
            document_dir = str(file_path_obj.parent)

            print(f"Загружаем документ: {document_name}")

            document = DocumentFactory.create_document(
                document_name=document_name,
                document_dir=document_dir,
                extract_dir_name=file_path_obj.stem,
                text_embeddings=self.text_embeddings,
                rebuild_index=self.rebuild_all_indexes
            )

            if document:
                print(f"✓ Успешно загружен: {document_name}")
                return document
            else:
                print(f"✗ Не удалось загрузить: {document_name}")
                return None

        except Exception as e:
            print(f"Ошибка при загрузке документа {file_path}: {e}")
            return None


# Пример использования
if __name__ == "__main__":
    print("Тестирование DocumentManager с динамическим масштабированием...")

    # Вариант с динамическим масштабированием
    document_manager = DocumentManager(
        rebuild_all_indexes=False,
        use_pretrained=True,
        use_word2vec=False,
        enable_dynamic_scaling=True
    )

    # Информация о загруженных документах
    info = document_manager.get_documents_info()
    print(f"Информация о документах: {info}")

    # Анализ доменов
    domain_analysis = document_manager.get_domain_analysis()
    print(f"Извлеченные домены: {list(domain_analysis.get('domains', {}).keys())}")

    # Пример 1: Поиск конкретного соответствия
    print("\n" + "=" * 50)
    print("ПОИСК КОНКРЕТНОГО СООТВЕТСТВИЯ")
    print("=" * 50)

    # Получаем список доступных файлов для демонстрации
    available_resumes = [doc.document_name for doc in document_manager.resume_documents[:]]
    available_vacancies = [doc.document_name for doc in document_manager.job_requirement_documents[:]]

    if available_resumes and available_vacancies:
        print(f"Доступные резюме: {available_resumes}")
        print(f"Доступные вакансии: {available_vacancies}")

        # Ищем соответствие для первого резюме и первой вакансии
        resume_name = available_resumes[0]
        vacancy_name = available_vacancies[0]

        match_result = document_manager.get_specific_match(resume_name, vacancy_name)

        if match_result:
            print(f"\nРЕЗУЛЬТАТ СООТВЕТСТВИЯ:")
            print(f"Резюме: {match_result['resume_name']}")
            print(f"Вакансия: {match_result['vacancy_name']}")
            print(f"Схожесть: {match_result['similarity_score']:.3f}")
            print(f"Уровень: {match_result['match_level']}")
            print(f"Рекомендация: {match_result['recommendation']}")

            # Детальная информация
            if match_result['domain_analysis']:
                print(f"\nДОМЕННЫЙ АНАЛИЗ:")
                print(f"Общие ключевые слова: {match_result['domain_analysis'].get('shared_keywords', [])[:5]}")
                print(
                    f"Отсутствующие ключевые слова: {match_result['domain_analysis'].get('missing_keywords', [])[:3]}")
        else:
            print("Не удалось найти соответствие")
    else:
        print("Недостаточно документов для тестирования")

    # Пример 2: Массовое сопоставление
    print("\n" + "=" * 50)
    print("МАССОВОЕ СОПОСТАВЛЕНИЕ КАНДИДАТОВ")
    print("=" * 50)

    matches = document_manager.match_candidates_to_job(top_n=5)

    if matches:
        for i, (score, candidate_info) in enumerate(matches, 1):
            print(f"{i}. Схожесть: {score:.3f}")
            print(f"   Кандидат: {candidate_info['metadata']['document_name']}")
            print(f"   Вакансия: {candidate_info['job_name']}")
            print("   " + "-" * 40)
    else:
        print("Не найдено совпадений")

    # Пример 3: Тестирование с разными комбинациями
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ РАЗНЫХ КОМБИНАЦИЙ")
    print("=" * 50)

    if len(available_resumes) > 1 and len(available_vacancies) > 1:
        # Тестируем несколько комбинаций
        test_combinations = [
            (available_resumes[0], available_vacancies[0]),
            (available_resumes[0], available_vacancies[1]),
            (available_resumes[1], available_vacancies[0])
        ]

        for resume, vacancy in test_combinations:
            result = document_manager.get_specific_match(resume, vacancy)
            if result:
                print(f"{resume} -> {vacancy}: {result['similarity_score']:.3f} ({result['match_level']})")

