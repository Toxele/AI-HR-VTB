import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
from typing import List, Dict
import os
import logging
from collections import Counter

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка стоп-слов и токенизатора
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


class Word2VecProcessor:
    def __init__(self, vector_size=300, window=3, min_count=2, workers=4):  # Увеличиваем vector_size, уменьшаем window
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        try:
            self.russian_stopwords = set(stopwords.words('russian'))
            self.english_stopwords = set(stopwords.words('english'))
        except:
            self.russian_stopwords = set()
            self.english_stopwords = set()

        # Дополнительные стоп-слова для IT-текстов
        self.it_stopwords = {
            'опыт', 'работа', 'разработка', 'проект', 'система', 'технология',
            'использование', 'время', 'компания', 'задача', 'решение', 'программа',
            'код', 'приложение', 'сервер', 'клиент', 'данные', 'информация'
        }

    def preprocess_text(self, text: str, keep_special_words: bool = False) -> List[str]:
        """Предобработка текста с улучшенной фильтрацией"""
        if not text:
            return []

        # Приводим к нижнему регистру
        text = text.lower()

        # Сохраняем специальные слова (скилы, технологии) перед очисткой
        special_words = set()
        if keep_special_words:
            # Ищем технологии, языки программирования, фреймворки
            tech_patterns = [
                r'python', r'java', r'javascript', r'sql', r'html', r'css',
                r'react', r'angular', r'vue', r'docker', r'kubernetes', r'aws',
                r'c\+\+', r'c#', r'php', r'ruby', r'go', r'rust', r'typescript',
                r'node\.js', r'django', r'flask', r'spring', r'laravel',
                r'mysql', r'postgresql', r'mongodb', r'redis', r'nginx', r'apache',
                r'linux', r'windows', r'macos', r'git', r'jenkins', r'ansible'
            ]

            for pattern in tech_patterns:
                matches = re.findall(pattern, text)
                special_words.update(matches)

        # Удаляем специальные символы, но сохраняем точки в технологиях
        text = re.sub(r'[^а-яёa-z0-9\.\#\+\s]', ' ', text)

        try:
            # Токенизация
            tokens = word_tokenize(text, language='russian')
        except:
            # Фолбэк: простая токенизация по пробелам
            tokens = text.split()

        # Удаляем стоп-слова и короткие слова
        filtered_tokens = []
        for token in tokens:
            # Пропускаем стоп-слова
            if (token in self.russian_stopwords or
                    token in self.english_stopwords or
                    token in self.it_stopwords):
                continue

            # Пропускаем слишком короткие слова (кроме технологий)
            if len(token) <= 2 and token not in special_words:
                continue

            # Пропускаем чисто числовые токены
            if token.isdigit():
                continue

            filtered_tokens.append(token)

        # Добавляем специальные слова обратно
        if keep_special_words:
            filtered_tokens.extend(special_words)

        return filtered_tokens

    def train_model(self, documents: List[str]):
        """Обучение модели с улучшенными параметрами"""
        if not documents:
            logger.warning("Нет документов для обучения")
            return

        logger.info("Подготовка данных для Word2Vec...")

        # Токенизируем все документы с сохранением специальных слов
        tokenized_docs = []
        for i, doc in enumerate(documents):
            if doc:
                tokens = self.preprocess_text(doc, keep_special_words=True)
                if tokens and len(tokens) >= 5:  # Минимум 5 токенов
                    tokenized_docs.append(tokens)

        if not tokenized_docs:
            logger.warning("Нет токенизированных данных для обучения")
            return

        logger.info(f"Обучение Word2Vec модели на {len(tokenized_docs)} документах...")

        try:
            # Улучшенные параметры обучения
            self.model = Word2Vec(
                sentences=tokenized_docs,
                vector_size=self.vector_size,  # Большая размерность
                window=self.window,  # Меньшее окно для лучшей специфичности
                min_count=self.min_count,
                workers=self.workers,
                sg=1,  # Skip-gram
                epochs=50,  # Больше эпох
                sample=1e-5,  # Subsampling частых слов
                negative=15,  # Negative sampling
                hs=0,  # No hierarchical softmax
                alpha=0.025,  # Learning rate
                min_alpha=0.0001  # Minimum learning rate
            )

            vocab_size = len(self.model.wv.key_to_index)
            logger.info(f"Модель Word2Vec обучена. Размер словаря: {vocab_size}")

            # Анализ качества модели
            self._analyze_model_quality(tokenized_docs)

        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            self.model = None

    def _analyze_model_quality(self, tokenized_docs):
        """Анализ качества обученной модели"""
        if not self.model:
            return

        # Проверяем семантическую связность
        test_words = ['python', 'java', 'разработчик', 'база', 'данные']
        print("\nКачество модели Word2Vec:")

        for word in test_words:
            if word in self.model.wv:
                try:
                    similar = self.model.wv.most_similar(word, topn=3)
                    print(f"  '{word}': {similar}")
                except:
                    print(f"  '{word}': не удалось найти похожие слова")

    def get_document_vector(self, text: str, method: str = 'weighted') -> np.ndarray:
        """Улучшенное получение векторного представления документа"""
        if self.model is None:
            return np.zeros(self.vector_size)

        tokens = self.preprocess_text(text, keep_special_words=True)
        if not tokens:
            return np.zeros(self.vector_size)

        if method == 'weighted':
            # Взвешенное среднее по IDF-подобным весам
            return self._get_weighted_average_vector(tokens)
        else:
            # Простое среднее
            return self._get_simple_average_vector(tokens)

    def _get_simple_average_vector(self, tokens: List[str]) -> np.ndarray:
        """Простое среднее векторов"""
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])

        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)

    def _get_weighted_average_vector(self, tokens: List[str]) -> np.ndarray:
        """Взвешенное среднее с учетом важности слов"""
        if not tokens:
            return np.zeros(self.vector_size)

        # Подсчитываем частоту слов (простой аналог IDF)
        word_counts = Counter(tokens)
        total_words = len(tokens)

        vectors = []
        weights = []

        for token in set(tokens):  # Уникальные токены
            if token in self.model.wv:
                # Вес = 1 / частота (редкие слова получают больший вес)
                frequency = word_counts[token] / total_words
                weight = 1.0 / (frequency + 0.001)  # Добавляем сглаживание

                vectors.append(self.model.wv[token])
                weights.append(weight)

        if vectors:
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Нормализуем веса
            return np.average(vectors, axis=0, weights=weights)
        else:
            return np.zeros(self.vector_size)

    def calculate_similarity(self, text1: str, text2: str, method: str = 'weighted') -> float:
        """Улучшенное вычисление схожести"""
        vec1 = self.get_document_vector(text1, method)
        vec2 = self.get_document_vector(text2, method)

        # Проверяем на нулевые векторы
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Косинусная схожесть с проверкой
        try:
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)

            # Нормализуем и ограничиваем (Word2Vec может давать >1 или < -1)
            cosine_sim = max(-1.0, min(1.0, cosine_sim))

            # Преобразуем в диапазон [0, 1]
            normalized_sim = (cosine_sim + 1) / 2

            # Применяем нелинейное преобразование для лучшего разделения
            if normalized_sim > 0.8:
                # "Сжимаем" очень высокие значения
                normalized_sim = 0.8 + (normalized_sim - 0.8) * 0.2

            return float(normalized_sim)
        except:
            return 0.0

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Семантическая схожесть с учетом ключевых терминов"""
        base_similarity = self.calculate_similarity(text1, text2, 'weighted')

        # Дополнительная проверка на совпадение ключевых терминов
        tokens1 = self.preprocess_text(text1, keep_special_words=True)
        tokens2 = self.preprocess_text(text2, keep_special_words=True)

        # Считаем совпадение ключевых терминов
        key_terms1 = set(tokens1) & self._get_technical_terms()
        key_terms2 = set(tokens2) & self._get_technical_terms()

        if key_terms1 and key_terms2:
            term_overlap = len(key_terms1 & key_terms2) / len(key_terms1 | key_terms2)
            # Комбинируем с базовой схожестью
            combined_similarity = 0.7 * base_similarity + 0.3 * term_overlap
            return combined_similarity

        return base_similarity

    def _get_technical_terms(self) -> set:
        """Возвращает набор технических терминов"""
        return {
            'python', 'java', 'javascript', 'sql', 'html', 'css', 'react',
            'angular', 'vue', 'docker', 'kubernetes', 'aws', 'c++', 'c#',
            'php', 'ruby', 'go', 'rust', 'typescript', 'node.js', 'django',
            'flask', 'spring', 'laravel', 'mysql', 'postgresql', 'mongodb',
            'redis', 'nginx', 'apache', 'linux', 'windows', 'macos', 'git'
        }

    def find_similar_words(self, word: str, top_n: int = 10):
        """Поиск семантически близких слов"""
        if self.model is None:
            return []

        if word in self.model.wv:
            try:
                return self.model.wv.most_similar(word, topn=top_n)
            except:
                return []
        else:
            return []

    def save_model(self, path: str):
        """Сохранение модели"""
        if self.model:
            self.model.save(path)
            logger.info(f"Модель сохранена в {path}")

    def load_model(self, path: str):
        """Загрузка модели"""
        if os.path.exists(path):
            try:
                self.model = Word2Vec.load(path)
                logger.info(f"Модель загружена из {path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                self.model = None
        else:
            logger.warning(f"Файл модели не найден: {path}")

    def get_vocabulary_size(self) -> int:
        """Получение размера словаря"""
        if self.model:
            return len(self.model.wv.key_to_index)
        return 0


# Пример использования
if __name__ == "__main__":
    # Тестовые данные
    sample_docs = [
        "Python разработчик с опытом работы Django Flask веб приложения",
        "Java программист Spring Hibernate базы данных MySQL PostgreSQL",
        "Бизнес аналитик требования проекты документация анализ данных",
        "Data Scientist машинное обучение Python анализ данных статистика"
    ]

    processor = Word2VecProcessor()
    processor.train_model(sample_docs)

    # Тестирование улучшенной схожести
    text1 = "Python разработчик Django Flask"
    text2 = "Java программист Spring Hibernate"
    text3 = "Python анализ данных машинное обучение"

    sim12 = processor.calculate_semantic_similarity(text1, text2)
    sim13 = processor.calculate_semantic_similarity(text1, text3)

    print(f"Схожесть (Python-Java): {sim12:.3f}")
    print(f"Схожесть (Python-DataScience): {sim13:.3f}")