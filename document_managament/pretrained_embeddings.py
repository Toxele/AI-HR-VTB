import numpy as np
from typing import List, Optional, Dict, Set, Any
import logging
import re
from collections import Counter
import math

logger = logging.getLogger(__name__)


class PretrainedEmbeddings:
    def __init__(self, model_type: str = "word2vec"):
        self.model = None
        self.model_type = model_type
        self.vector_size = 300
        self.is_loaded = False
        self.fallback_embeddings = {}
        self.domain_keywords = self._load_domain_keywords()
        self.idf_weights = {}
        self.dynamic_weights = {}  # Для динамических весов
        self.load_model()

    def _load_domain_keywords(self) -> Dict[str, float]:
        """Ключевые слова для IT/HR домена с весами"""
        return {
            # IT Skills & Technologies
            'python': 2.0, 'java': 2.0, 'javascript': 2.0, 'sql': 1.8,
            'html': 1.5, 'css': 1.5, 'react': 1.8, 'angular': 1.8,
            'vue': 1.8, 'docker': 1.7, 'kubernetes': 1.7, 'aws': 1.7,
            'django': 1.8, 'flask': 1.8, 'spring': 1.8, 'hibernate': 1.7,
            'postgresql': 1.6, 'mysql': 1.6, 'mongodb': 1.6, 'redis': 1.6,
            'nginx': 1.5, 'apache': 1.5, 'linux': 1.6, 'git': 1.5,

            # Job Roles & Positions
            'разработчик': 1.8, 'программист': 1.8, 'аналитик': 2.0,
            'архитектор': 1.7, 'инженер': 1.6, 'специалист': 1.5,
            'менеджер': 1.6, 'лид': 1.7, 'тимлид': 1.8,

            # Business & HR Terms
            'бизнес': 1.8, 'анализ': 1.9, 'требования': 1.9,
            'проект': 1.6, 'система': 1.6, 'технология': 1.5,
            'вакансия': 1.7, 'резюме': 1.7, 'собеседование': 1.6,
            'опыт': 1.6, 'навыки': 1.7, 'компетенции': 1.6,

            # Specific Domain Terms
            'данные': 1.8, 'база': 1.7, 'backend': 1.7, 'frontend': 1.7,
            'fullstack': 1.7, 'devops': 1.7, 'qa': 1.6, 'тестирование': 1.6,
            'разработка': 1.6, 'программирование': 1.6, 'код': 1.5
        }

    def update_domain_weights(self, new_weights: Dict[str, float]):
        """Обновление весов доменных ключевых слов"""
        for word, weight in new_weights.items():
            self.domain_keywords[word] = weight
            # Также обновляем IDF веса
            self.idf_weights[word] = weight

        print(f"Обновлено весов: {len(new_weights)}")

    def load_model(self):
        """Загрузка предобученной модели"""
        try:
            print("Попытка загрузки предобученных эмбеддингов...")

            if self.model_type == "word2vec":
                self._load_word2vec_model()
            elif self.model_type == "fasttext":
                self._load_fasttext_model()
            else:
                self._load_fallback_embeddings()

        except Exception as e:
            print(f"Ошибка загрузки предобученной модели: {e}")
            self._load_fallback_embeddings()

    def _load_word2vec_model(self):
        """Загрузка Word2Vec модели"""
        try:
            import gensim.downloader as api
            print("Загрузка Word2Vec модели (ruscorpora)...")
            self.model = api.load("word2vec-ruscorpora-300")
            self.vector_size = 300
            self.is_loaded = True
            print(f"Word2Vec модель загружена. Словарь: {len(self.model.key_to_index)} слов")

            # Создаем кэш для быстрого поиска слов без тегов
            self._build_word_cache()
            # Инициализируем IDF веса
            self._initialize_idf_weights()

        except Exception as e:
            print(f"Не удалось загрузить Word2Vec: {e}")
            self._load_fallback_embeddings()

    def _build_word_cache(self):
        """Создание кэша для поиска слов без тегов POS"""
        self.word_cache = {}
        for word_with_tag in self.model.key_to_index.keys():
            # Извлекаем базовое слово без тега (python_NOUN -> python)
            base_word = word_with_tag.split('_')[0].lower()
            if base_word not in self.word_cache:
                self.word_cache[base_word] = []
            self.word_cache[base_word].append(word_with_tag)

        # Добавляем популярные английские слова в разных форматах
        self._add_common_english_words()

    def _initialize_idf_weights(self):
        """Инициализация IDF весов для ключевых слов"""
        # Базовые IDF веса (можно настроить на реальных данных)
        for word, weight in self.domain_keywords.items():
            self.idf_weights[word] = weight

        # Добавляем веса для слов с тегами
        for word_with_tag in self.model.key_to_index.keys():
            base_word = word_with_tag.split('_')[0].lower()
            if base_word in self.domain_keywords:
                self.idf_weights[word_with_tag] = self.domain_keywords[base_word]

    def _add_common_english_words(self):
        """Добавляем распространенные английские слова технологий"""
        common_tech_words = [
            'python', 'django', 'flask', 'java', 'javascript', 'react',
            'angular', 'vue', 'docker', 'kubernetes', 'aws', 'sql',
            'html', 'css', 'spring', 'hibernate', 'postgresql', 'mysql',
            'mongodb', 'redis', 'nginx', 'apache', 'linux', 'git',
            'backend', 'frontend', 'fullstack', 'devops', 'qa'
        ]

        for word in common_tech_words:
            if word not in self.word_cache:
                # Пробуем найти слово с разными тегами
                possible_tags = ['_NOUN', '_VERB', '_ADJ', '_ADV', '_UNKN']
                for tag in possible_tags:
                    tagged_word = word + tag
                    if tagged_word in self.model.key_to_index:
                        if word not in self.word_cache:
                            self.word_cache[word] = []
                        self.word_cache[word].append(tagged_word)
                        break

    def _load_fasttext_model(self):
        """Загрузка FastText модели"""
        try:
            import gensim.downloader as api
            print("Загрузка FastText модели...")
            self.model = api.load("fasttext-wiki-news-subwords-300")
            self.vector_size = 300
            self.is_loaded = True
            print(f"FastText модель загружена. Словарь: {len(self.model.key_to_index)} слов")
        except Exception as e:
            print(f"Не удалось загрузить FastText: {e}")
            self._load_fallback_embeddings()

    def _load_fallback_embeddings(self):
        """Создание fallback эмбеддингов"""
        print("Использование fallback эмбеддингов...")
        self.model = None
        self.vector_size = 300
        self.is_loaded = False

        # Создаем простой словарь базовых слов
        basic_words = list(self.domain_keywords.keys())
        for word in basic_words:
            self.fallback_embeddings[word] = np.random.randn(self.vector_size)

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Получение вектора слова"""
        word_lower = word.lower()

        if self.model is not None:
            # Пробуем найти слово в кэше
            if hasattr(self, 'word_cache') and word_lower in self.word_cache:
                tagged_words = self.word_cache[word_lower]
                # Берем первый доступный вариант
                for tagged_word in tagged_words:
                    if tagged_word in self.model.key_to_index:
                        return self.model[tagged_word]

            # Пробуем найти слово как есть (на случай английских слов)
            if word_lower in self.model.key_to_index:
                return self.model[word_lower]

            # Пробуем найти слово с разными тегами
            possible_tags = ['_NOUN', '_VERB', '_ADJ', '_ADV', '_UNKN']
            for tag in possible_tags:
                tagged_word = word_lower + tag
                if tagged_word in self.model.key_to_index:
                    return self.model[tagged_word]

        # Fallback на случайные эмбеддинги
        if word_lower in self.fallback_embeddings:
            return self.fallback_embeddings[word_lower]

        return None

    def get_document_vector(self, text: str, method: str = 'weighted_domain') -> np.ndarray:
        """Получение вектора документа с домен-специфичными весами"""
        if not text:
            return np.zeros(self.vector_size)

        words = self._preprocess_text(text)
        vectors = []
        weights = []
        missing_words = []

        for word in words:
            vec = self.get_word_vector(word)
            if vec is not None:
                vectors.append(vec)
                # Вес слова based on domain importance
                weight = self._get_word_weight(word)
                weights.append(weight)
            else:
                missing_words.append(word)

        # Если есть пропущенные слова, создаем для них случайные векторы
        if missing_words and not vectors:
            for word in missing_words:
                random_vec = np.random.randn(self.vector_size)
                vectors.append(random_vec)
                weights.append(1.0)  # Базовый вес
                self.fallback_embeddings[word.lower()] = random_vec

        if not vectors:
            return np.zeros(self.vector_size)

        if method == 'weighted_domain':
            return self._get_weighted_domain_vector(vectors, weights)
        elif method == 'average':
            return np.mean(vectors, axis=0)
        else:
            return np.mean(vectors, axis=0)

    def _get_word_weight(self, word: str) -> float:
        """Получение веса слова based on domain importance"""
        word_lower = word.lower()

        # Проверяем доменные ключевые слова
        if word_lower in self.domain_keywords:
            return self.domain_keywords[word_lower]

        # Проверяем IDF веса
        if word_lower in self.idf_weights:
            return self.idf_weights[word_lower]

        # Для слов с тегами
        if hasattr(self, 'word_cache') and word_lower in self.word_cache:
            for tagged_word in self.word_cache[word_lower]:
                if tagged_word in self.idf_weights:
                    return self.idf_weights[tagged_word]

        # Базовый вес для остальных слов
        return 1.0

    def _get_weighted_domain_vector(self, vectors: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Взвешенное среднее с домен-специфичными весами"""
        weights = np.array(weights)

        # Нормализуем веса
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)

        return np.average(vectors, axis=0, weights=weights)

    def _preprocess_text(self, text: str) -> List[str]:
        """Предобработка текста с выделением ключевых фраз"""
        if not text:
            return []

        # Приводим к нижнему регистру
        text = text.lower()

        # Удаляем специальные символы, оставляем буквы, цифры и пробелы
        text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)

        # Разбиваем на слова
        words = text.split()

        # Удаляем короткие слова и стоп-слова
        words = self._filter_words(words)

        return words

    def _filter_words(self, words: List[str]) -> List[str]:
        """Фильтрация слов с учетом доменной специфики"""
        stop_words = {
            'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'за',
            'к', 'у', 'о', 'об', 'не', 'что', 'как', 'так', 'это', 'то',
            'все', 'его', 'ее', 'их', 'им', 'ими', 'него', 'нее', 'них',
            'который', 'которая', 'которые', 'которого', 'которой', 'которых'
        }

        filtered_words = []
        for word in words:
            if len(word) > 2 and word not in stop_words:
                filtered_words.append(word)

        return filtered_words

    def calculate_similarity(self, text1: str, text2: str, enhance_discrimination: bool = True) -> float:
        """Вычисление косинусной схожести с усилением дискриминативности"""
        vec1 = self.get_document_vector(text1, 'weighted_domain')
        vec2 = self.get_document_vector(text2, 'weighted_domain')

        # Добавим небольшую константу чтобы избежать деления на ноль
        norm1 = np.linalg.norm(vec1) + 1e-10
        norm2 = np.linalg.norm(vec2) + 1e-10

        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)

        # Нормализуем в диапазон [0, 1]
        base_similarity = max(0.0, min(1.0, (cosine_sim + 1) / 2))

        if enhance_discrimination:
            return self._enhance_discrimination(base_similarity, text1, text2)
        else:
            return base_similarity

    def _enhance_discrimination(self, base_similarity: float, text1: str, text2: str) -> float:
        """Усиление дискриминативности схожести"""
        # 1. Проверка совпадения ключевых терминов
        key_terms_match = self._calculate_key_terms_match(text1, text2)

        # 2. Проверка доменной релевантности
        domain_relevance = self._calculate_domain_relevance(text1, text2)

        # 3. Нелинейное преобразование для усиления различий
        enhanced_similarity = self._nonlinear_transform(
            base_similarity,
            key_terms_match,
            domain_relevance
        )

        return enhanced_similarity

    def _calculate_key_terms_match(self, text1: str, text2: str) -> float:
        """Вычисление совпадения ключевых терминов"""
        words1 = set(self._preprocess_text(text1))
        words2 = set(self._preprocess_text(text2))

        # Ключевые термины из обоих текстов
        key_terms1 = words1 & set(self.domain_keywords.keys())
        key_terms2 = words2 & set(self.domain_keywords.keys())

        if not key_terms1 or not key_terms2:
            return 0.0

        # Совпадение ключевых терминов
        matched_terms = key_terms1 & key_terms2

        # Взвешенное совпадение
        total_weight = sum(self.domain_keywords[term] for term in key_terms1 | key_terms2)
        matched_weight = sum(self.domain_keywords[term] for term in matched_terms)

        if total_weight > 0:
            return matched_weight / total_weight
        return 0.0

    def _calculate_domain_relevance(self, text1: str, text2: str) -> float:
        """Вычисление доменной релевантности"""
        words1 = self._preprocess_text(text1)
        words2 = self._preprocess_text(text2)

        # Количество ключевых слов в каждом текстах
        key_words1 = [w for w in words1 if w in self.domain_keywords]
        key_words2 = [w for w in words2 if w in self.domain_keywords]

        # Релевантность based on key words presence
        relevance1 = len(key_words1) / len(words1) if words1 else 0
        relevance2 = len(key_words2) / len(words2) if words2 else 0

        return min(relevance1, relevance2)  # Минимальная релевантность

    def _nonlinear_transform(self, base_sim: float, key_terms: float, domain_rel: float) -> float:
        """Нелинейное преобразование для усиления дискриминативности"""
        # Усиливаем влияние ключевых терминов
        if key_terms > 0.7:  # Хорошее совпадение терминов
            enhanced_sim = base_sim * 0.6 + key_terms * 0.4
        else:
            enhanced_sim = base_sim

        # Учет доменной релевантности
        if domain_rel < 0.3:  # Низкая релевантность
            enhanced_sim *= 0.7  # Сильно понижаем

        # Нелинейное сжатие для высоких значений
        if enhanced_sim > 0.8:
            enhanced_sim = 0.8 + (enhanced_sim - 0.8) * 0.3

        # Нелинейное растяжение для низких значений
        if enhanced_sim < 0.4:
            enhanced_sim = enhanced_sim * 1.2

        return max(0.0, min(1.0, enhanced_sim))

    def is_model_loaded(self) -> bool:
        """Проверка, загружена ли модель"""
        return self.is_loaded

    def get_vocabulary_size(self) -> int:
        """Получение размера словаря"""
        if self.model is not None:
            return len(self.model.key_to_index)
        return len(self.fallback_embeddings)

    def _calculate_vacancy_specific_similarity(self, job_text: str, resume_text: str, job_title: str) -> float:
        """Вычисление схожести с учетом специфики вакансии"""
        base_similarity = self.calculate_similarity(job_text, resume_text)

        vacancy_type = self._classify_vacancy_type(job_title)
        resume_type = self._classify_resume_type(resume_text)
        adjustment = self._get_similarity_adjustment(vacancy_type, resume_type)

        final_similarity = base_similarity * adjustment

        # Логгирование для отладки
        print(f"  Вакансия: {vacancy_type}, Резюме: {resume_type}, "
              f"Базовая: {base_similarity:.3f}, Коррекция: {adjustment:.2f}, "
              f"Итог: {final_similarity:.3f}")

        return final_similarity

    def calculate_vacancy_specific_similarity(self, job_text: str, resume_text: str, job_title: str) -> float:
        """Публичный метод для вакансио-специфичной схожести"""
        return self._calculate_vacancy_specific_similarity(job_text, resume_text, job_title)

    def _classify_vacancy_type(self, job_title: str) -> str:
        """Классификация типа вакансии"""
        job_lower = job_title.lower()

        if any(word in job_lower for word in ['бизнес', 'аналитик', 'анализ', 'данн']):
            return 'business_analyst'
        elif any(word in job_lower for word in ['разработчик', 'программист', 'developer', 'engineer']):
            return 'developer'
        elif any(word in job_lower for word in ['ит', 'информацион', 'систем', 'техническ']):
            return 'it_general'
        else:
            return 'general'

    def _classify_resume_type(self, resume_text: str) -> str:
        """Классификация типа резюме"""
        text_lower = resume_text.lower()

        # Считаем ключевые слова для каждого типа
        business_keywords = ['бизнес', 'анализ', 'данн', 'требован', 'проект']
        developer_keywords = ['разработ', 'программир', 'код', 'python', 'java', 'javascript']
        it_keywords = ['систем', 'техническ', 'администрир', 'сеть', 'база']

        business_score = sum(1 for word in business_keywords if word in text_lower)
        developer_score = sum(1 for word in developer_keywords if word in text_lower)
        it_score = sum(1 for word in it_keywords if word in text_lower)

        scores = {
            'business_analyst': business_score,
            'developer': developer_score,
            'it_general': it_score
        }

        # Возвращаем тип с наибольшим score
        return max(scores.items(), key=lambda x: x[1])[0]

    def _get_similarity_adjustment(self, vacancy_type: str, resume_type: str) -> float:
        """Корректировка схожести based on type matching"""
        adjustment_matrix = {
            'business_analyst': {
                'business_analyst': 1.3,  # Увеличиваем для релевантных
                'developer': 0.7,  # Уменьшаем для нерелевантных
                'it_general': 0.8,
                'general': 1.0
            },
            'developer': {
                'business_analyst': 0.7,
                'developer': 1.3,
                'it_general': 1.1,
                'general': 1.0
            },
            'it_general': {
                'business_analyst': 0.8,
                'developer': 1.1,
                'it_general': 1.2,
                'general': 1.0
            },
            'general': {
                'business_analyst': 1.0,
                'developer': 1.0,
                'it_general': 1.0,
                'general': 1.0
            }
        }

        return adjustment_matrix.get(vacancy_type, {}).get(resume_type, 1.0)

    def calculate_similarity_detailed(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Детальный расчет схожести с дополнительной информацией

        Returns:
            Словарь с детальной информацией о схожести
        """
        vec1 = self.get_document_vector(text1, 'weighted_domain')
        vec2 = self.get_document_vector(text2, 'weighted_domain')

        # Косинусная схожесть
        norm1 = np.linalg.norm(vec1) + 1e-10
        norm2 = np.linalg.norm(vec2) + 1e-10
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        base_similarity = max(0.0, min(1.0, (cosine_sim + 1) / 2))

        # Улучшенная схожесть
        enhanced_similarity = self._enhance_discrimination(base_similarity, text1, text2)

        # Анализ ключевых слов
        keywords_analysis = self._analyze_keywords_match(text1, text2)

        return {
            'cosine_similarity': float(base_similarity),
            'enhanced_similarity': float(enhanced_similarity),
            'keywords_analysis': keywords_analysis,
            'vector_norm_text1': float(norm1),
            'vector_norm_text2': float(norm2)
        }

    def _analyze_keywords_match(self, text1: str, text2: str) -> Dict[str, Any]:
        """Детальный анализ совпадения ключевых слов"""
        words1 = set(self._preprocess_text(text1))
        words2 = set(self._preprocess_text(text2))

        # Все ключевые слова
        all_keywords = set(self.domain_keywords.keys())

        # Ключевые слова в каждом тексте
        keywords1 = words1 & all_keywords
        keywords2 = words2 & all_keywords

        # Совпадающие ключевые слова
        matched_keywords = keywords1 & keywords2

        # Уникальные ключевые слова
        unique1 = keywords1 - keywords2
        unique2 = keywords2 - keywords1

        # Взвешенные оценки
        matched_weight = sum(self.domain_keywords[word] for word in matched_keywords)
        total_weight = sum(self.domain_keywords[word] for word in (keywords1 | keywords2))

        coverage = matched_weight / total_weight if total_weight > 0 else 0

        return {
            'matched_keywords': list(matched_keywords),
            'unique_text1': list(unique1),
            'unique_text2': list(unique2),
            'coverage_score': float(coverage),
            'matched_count': len(matched_keywords),
            'total_keywords_count': len(keywords1 | keywords2)
        }


# Пример использования
if __name__ == "__main__":
    embeddings = PretrainedEmbeddings()

    # Тестируем отдельные слова
    test_words = ['python', 'java', 'разработка', 'данные', 'django', 'flask', 'spring']
    print("Тестирование отдельных слов:")
    for word in test_words:
        vec = embeddings.get_word_vector(word)
        if vec is not None:
            print(f"  '{word}': найден")
        else:
            print(f"  '{word}': не найден")

    text1 = "Python разработчик Django Flask"
    text2 = "Java программист Spring Hibernate"
    text3 = "Python анализ данных"
    text4 = "веб разработка JavaScript React"
    text5 = "база данных SQL PostgreSQL"

    print("\nСравнение текстов:")
    sim12 = embeddings.calculate_similarity(text1, text2)
    sim13 = embeddings.calculate_similarity(text1, text3)
    sim14 = embeddings.calculate_similarity(text1, text4)
    sim15 = embeddings.calculate_similarity(text1, text5)

    print(f"Python разработчик vs Java программист: {sim12:.3f}")
    print(f"Python разработчик vs Python анализ данных: {sim13:.3f}")
    print(f"Python разработчик vs веб разработка: {sim14:.3f}")
    print(f"Python разработчик vs база данных: {sim15:.3f}")

    print(f"\nМодель загружена: {embeddings.is_model_loaded()}")
    print(f"Размер словаря: {embeddings.get_vocabulary_size()}")