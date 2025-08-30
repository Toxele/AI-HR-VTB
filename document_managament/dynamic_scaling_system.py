import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
import re
from collections import Counter, defaultdict
import math
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

logger = logging.getLogger(__name__)

try:
    nltk.download('stopwords', quiet=True)
except:
    pass


class DynamicScalingSystem:
    def __init__(self,
                 min_domain_frequency: int = 3,
                 min_subdomain_frequency: int = 2,
                 max_domains: int = 20,
                 max_subdomains_per_domain: int = 10):

        self.min_domain_frequency = min_domain_frequency
        self.min_subdomain_frequency = min_subdomain_frequency
        self.max_domains = max_domains
        self.max_subdomains_per_domain = max_subdomains_per_domain

        self.domains = {}
        self.subdomains = {}
        self.domain_keywords = {}
        self.subdomain_keywords = {}
        self.word_weights = {}
        self.document_frequencies = Counter()

        # Инициализация стоп-слов и стеммера
        self.stop_words = set(stopwords.words('russian') + stopwords.words('english'))
        self.stemmer = SnowballStemmer('russian')

        # Предопределенные паттерны для IT домена
        self.domain_patterns = {
            'development': {
                'keywords': ['python', 'java', 'javascript', 'разработка', 'программирование', 'код'],
                'patterns': [r'.*разработчик.*', r'.*программист.*', r'.*developer.*']
            },
            'data_science': {
                'keywords': ['данные', 'анализ', 'ml', 'ai', 'машинное', 'обучение'],
                'patterns': [r'.*аналитик.*', r'.*data.*scientist.*', r'.*ml.*engineer.*']
            },
            'devops': {
                'keywords': ['docker', 'kubernetes', 'aws', 'azure', 'devops', 'ci/cd'],
                'patterns': [r'.*devops.*', r'.*инженер.*инфраструктур.*', r'.*sre.*']
            },
            'testing': {
                'keywords': ['тестирование', 'qa', 'автотесты', 'quality', 'assurance'],
                'patterns': [r'.*тестировщик.*', r'.*qa.*engineer.*', r'.*quality.*assurance.*']
            }
        }

    def analyze_documents(self, documents: List[str], document_types: List[str] = None):
        """Анализ документов для извлечения доменов и поддоменов"""
        if not documents:
            return

        print(f"Анализ {len(documents)} документов для динамического извлечения доменов...")

        # Предобработка текстов
        processed_texts = [self._preprocess_text(doc) for doc in documents]

        # Извлечение ключевых терминов
        self._extract_key_terms(processed_texts)

        # Кластеризация для выявления доменов
        self._cluster_domains(processed_texts)

        # Извлечение поддоменов
        self._extract_subdomains(processed_texts)

        # Расчет весов слов
        self._calculate_word_weights(processed_texts)

        print(f"Извлечено {len(self.domains)} доменов и {len(self.subdomains)} поддоменов")

    def _preprocess_text(self, text: str) -> List[str]:
        """Предобработка текста"""
        if not text:
            return []

        # Приведение к нижнему регистру
        text = text.lower()

        # Удаление специальных символов
        text = re.sub(r'[^а-яёa-z0-9\s\-]', ' ', text)

        # Токенизация
        words = text.split()

        # Удаление стоп-слов и коротких слов
        words = [word for word in words if word not in self.stop_words and len(word) > 2]

        # Стемминг (для русского языка)
        words = [self.stemmer.stem(word) if self._is_russian(word) else word for word in words]

        return words

    def _is_russian(self, word: str) -> bool:
        """Проверка, является ли слово русским"""
        return any('а' <= char <= 'я' or char == 'ё' for char in word.lower())

    def _extract_key_terms(self, processed_texts: List[List[str]]):
        """Извлечение ключевых терминов из текстов"""
        # Собираем статистику по всем словам
        all_words = []
        for words in processed_texts:
            all_words.extend(words)

        word_freq = Counter(all_words)

        # Фильтруем по минимальной частоте
        self.key_terms = {word: freq for word, freq in word_freq.items()
                          if freq >= self.min_domain_frequency}

        # Добавляем N-граммы (биграммы и триграммы)
        self._extract_ngrams(processed_texts)

    def _extract_ngrams(self, processed_texts: List[List[str]], n: int = 3):
        """Извлечение N-грамм"""
        for words in processed_texts:
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i + n])
                if ngram not in self.key_terms:
                    self.key_terms[ngram] = 1
                else:
                    self.key_terms[ngram] += 1

    def _cluster_domains(self, processed_texts: List[List[str]]):
        """Кластеризация текстов для выявления доменов"""
        # Создаем TF-IDF матрицу
        text_strings = [' '.join(words) for words in processed_texts]

        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.8,
                stop_words=list(self.stop_words)
            )

            tfidf_matrix = vectorizer.fit_transform(text_strings)

            # Кластеризация с помощью K-means
            n_clusters = min(self.max_domains, max(2, len(processed_texts) // 10))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(tfidf_matrix)

                # Анализ кластеров для определения доменов
                self._analyze_clusters(clusters, processed_texts, vectorizer.get_feature_names_out())

        except Exception as e:
            print(f"Ошибка кластеризации: {e}")
            # Fallback: использование предопределенных паттернов
            self._use_predefined_domains(processed_texts)

    def _analyze_clusters(self, clusters, processed_texts, feature_names):
        """Анализ кластеров для определения доменов"""
        cluster_terms = defaultdict(Counter)

        for i, cluster_id in enumerate(clusters):
            for word in processed_texts[i]:
                cluster_terms[cluster_id][word] += 1

        # Определяем домены по топ-терминам в каждом кластере
        for cluster_id, term_counter in cluster_terms.items():
            top_terms = term_counter.most_common(10)
            domain_name = f"domain_{cluster_id}"

            # Пытаемся найти подходящее имя домена
            for term, count in top_terms:
                for predefined_domain, pattern_info in self.domain_patterns.items():
                    if term in pattern_info['keywords']:
                        domain_name = predefined_domain
                        break

            self.domains[domain_name] = {
                'terms': dict(top_terms),
                'size': sum(term_counter.values()),
                'cluster_id': cluster_id
            }

    def _use_predefined_domains(self, processed_texts):
        """Использование предопределенных доменов на основе паттернов"""
        domain_counts = defaultdict(int)

        for words in processed_texts:
            text = ' '.join(words)

            for domain_name, pattern_info in self.domain_patterns.items():
                for pattern in pattern_info['patterns']:
                    if re.match(pattern, text, re.IGNORECASE):
                        domain_counts[domain_name] += 1
                        break

                # Также проверяем по ключевым словам
                keyword_matches = sum(1 for keyword in pattern_info['keywords']
                                      if any(keyword in word for word in words))
                if keyword_matches >= 2:
                    domain_counts[domain_name] += 1

        # Создаем домены с достаточной частотой
        for domain_name, count in domain_counts.items():
            if count >= self.min_domain_frequency:
                self.domains[domain_name] = {
                    'terms': self.domain_patterns[domain_name]['keywords'],
                    'size': count,
                    'cluster_id': -1
                }

    def _extract_subdomains(self, processed_texts: List[List[str]]):
        """Извлечение поддоменов для каждого домена"""
        for domain_name in self.domains.keys():
            domain_texts = []

            # Собираем тексты, относящиеся к домену
            for words in processed_texts:
                text = ' '.join(words)
                if self._is_domain_related(text, domain_name):
                    domain_texts.append(words)

            if len(domain_texts) >= self.min_subdomain_frequency:
                self._cluster_subdomains(domain_name, domain_texts)

    def _is_domain_related(self, text: str, domain_name: str) -> bool:
        """Проверка, относится ли текст к домену"""
        if domain_name in self.domain_patterns:
            pattern_info = self.domain_patterns[domain_name]
            # Проверка по паттернам
            for pattern in pattern_info['patterns']:
                if re.match(pattern, text, re.IGNORECASE):
                    return True
            # Проверка по ключевым словам
            keyword_matches = sum(1 for keyword in pattern_info['keywords']
                                  if keyword in text)
            return keyword_matches >= 2

        # Для автоматически обнаруженных доменов
        domain_terms = self.domains[domain_name]['terms']
        term_matches = sum(1 for term in domain_terms if term in text)
        return term_matches >= 2

    def _cluster_subdomains(self, domain_name: str, domain_texts: List[List[str]]):
        """Кластеризация поддоменов для домена"""
        text_strings = [' '.join(words) for words in domain_texts]

        try:
            vectorizer = TfidfVectorizer(
                max_features=200,
                min_df=1,
                max_df=0.9
            )

            tfidf_matrix = vectorizer.fit_transform(text_strings)

            # Используем DBSCAN для обнаружения плотных областей
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            subclusters = dbscan.fit_predict(tfidf_matrix.toarray())

            unique_subclusters = set(subclusters)
            if -1 in unique_subclusters:  # Убираем шум
                unique_subclusters.remove(-1)

            for subcluster_id in unique_subclusters:
                if subcluster_id != -1:
                    subdomain_name = f"{domain_name}_sub_{subcluster_id}"
                    self.subdomains[subdomain_name] = {
                        'domain': domain_name,
                        'size': sum(1 for x in subclusters if x == subcluster_id)
                    }

        except Exception as e:
            print(f"Ошибка кластеризации поддоменов для {domain_name}: {e}")

    def _calculate_word_weights(self, processed_texts: List[List[str]]):
        """Расчет весов слов на основе TF-IDF и доменной релевантности"""
        # Сначала вычисляем стандартный TF-IDF
        all_documents = [' '.join(words) for words in processed_texts]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_documents)
        feature_names = vectorizer.get_feature_names_out()

        # IDF веса
        idf_weights = dict(zip(feature_names, vectorizer.idf_))

        # Доменная релевантность
        domain_relevance = self._calculate_domain_relevance(feature_names)

        # Комбинируем веса
        for word in feature_names:
            tfidf_weight = idf_weights.get(word, 1.0)
            domain_weight = domain_relevance.get(word, 1.0)

            # Комбинированный вес (логарифмическая комбинация)
            combined_weight = math.log1p(tfidf_weight) * math.log1p(domain_weight)
            self.word_weights[word] = combined_weight

    def _calculate_domain_relevance(self, feature_names) -> Dict[str, float]:
        """Расчет релевантности слов доменам"""
        domain_relevance = {}

        for word in feature_names:
            max_relevance = 0.0

            for domain_name, domain_info in self.domains.items():
                if word in domain_info['terms']:
                    # Вес based on частоте в домене
                    frequency = domain_info['terms'][word]
                    domain_size = domain_info['size']
                    relevance = frequency / domain_size

                    # Увеличиваем вес для специализированных терминов
                    if self._is_specialized_term(word, domain_name):
                        relevance *= 2.0

                    max_relevance = max(max_relevance, relevance)

            domain_relevance[word] = max_relevance

        return domain_relevance

    def _is_specialized_term(self, word: str, domain_name: str) -> bool:
        """Проверка, является ли термин специализированным"""
        # Специализированные термины обычно реже встречаются в общем языке
        specialized_indicators = [
            len(word) > 6,  # Длинные слова
            not self._is_common_word(word),  # Не common word
            any(char.isdigit() for char in word),  # Содержит цифры
            '-' in word or '_' in word  # Содержит разделители
        ]

        return sum(specialized_indicators) >= 2

    def _is_common_word(self, word: str) -> bool:
        """Проверка, является ли слово common word"""
        common_words = {
            'работа', 'опыт', 'навыки', 'знание', 'умение', 'обязанность',
            'требование', 'задача', 'проект', 'команда', 'разработка',
            'время', 'год', 'месяц', 'день', 'работать', 'делать', 'создавать'
        }
        return word in common_words

    def get_domains(self) -> Dict:
        """Возвращает извлеченные домены"""
        return self.domains

    def get_subdomains(self) -> Dict:
        """Возвращает извлеченные поддомены"""
        return self.subdomains

    def get_domain_weights(self) -> Dict[str, float]:
        """Возвращает веса доменных ключевых слов"""
        domain_weights = {}

        for domain_name, domain_info in self.domains.items():
            for term, frequency in domain_info['terms'].items():
                if term in self.word_weights:
                    # Нормализуем вес based on важности в домене
                    domain_size = domain_info['size']
                    normalized_weight = (frequency / domain_size) * self.word_weights[term]

                    # Увеличиваем вес для специализированных терминов
                    if self._is_specialized_term(term, domain_name):
                        normalized_weight *= 1.5

                    domain_weights[term] = normalized_weight

        return domain_weights

    def get_word_weight(self, word: str) -> float:
        """Возвращает вес конкретного слова"""
        return self.word_weights.get(word, 1.0)

    def suggest_domain_for_text(self, text: str) -> str:
        """Предсказывает домен для нового текста"""
        processed_text = self._preprocess_text(text)
        text_str = ' '.join(processed_text)

        best_domain = 'general'
        best_score = 0.0

        for domain_name, domain_info in self.domains.items():
            score = 0.0

            # Считаем совпадения с терминами домена
            for term in domain_info['terms']:
                if term in text_str:
                    score += self.get_word_weight(term)

            # Нормализуем по длине текста
            if processed_text:
                score /= len(processed_text)

            if score > best_score:
                best_score = score
                best_domain = domain_name

        return best_domain

    def update_with_new_data(self, new_texts: List[str], document_types: List[str] = None):
        """Обновление системы с новыми данными"""
        processed_new_texts = [self._preprocess_text(text) for text in new_texts]

        # Обновляем частоты терминов
        for words in processed_new_texts:
            for word in words:
                if word in self.key_terms:
                    self.key_terms[word] += 1
                else:
                    self.key_terms[word] = 1

        # Пересчитываем веса
        all_processed_texts = []
        for domain_info in self.domains.values():
            # Здесь нужно было бы пересобрать все тексты, но для простоты
            # мы просто добавляем новые данные к существующим весам
            pass

        # Пересчитываем веса слов
        self._calculate_word_weights(processed_new_texts)

        print(f"Система обновлена с {len(new_texts)} новыми документами")


# Пример использования
if __name__ == "__main__":
    print("Тестирование DynamicScalingSystem...")

    # Примерные тексты для тестирования
    test_documents = [
        "Python разработчик с опытом работы Django Flask",
        "Java программист Spring Hibernate базы данных",
        "Data Scientist машинное обучение анализ данных Python",
        "DevOps инженер Docker Kubernetes AWS CI/CD",
        "Frontend разработчик JavaScript React Vue HTML CSS",
        "Backend разработчик Python Django REST API",
        "Аналитик данных SQL Python бизнес анализ",
        "Системный администратор Linux сеть безопасность",
        "Тестировщик QA автоматическое тестирование Selenium",
        "Project менеджер управление проектами Agile Scrum"
    ]

    scaling_system = DynamicScalingSystem(
        min_domain_frequency=2,
        min_subdomain_frequency=1
    )

    scaling_system.analyze_documents(test_documents)

    print("\nИзвлеченные домены:")
    for domain_name, domain_info in scaling_system.get_domains().items():
        print(f"  {domain_name}: {list(domain_info['terms'].keys())[:5]}")

    print("\nВеса ключевых слов:")
    domain_weights = scaling_system.get_domain_weights()
    for word, weight in list(domain_weights.items())[:10]:
        print(f"  {word}: {weight:.3f}")

    # Тестирование предсказания домена
    test_text = "Python Django разработка веб приложений"
    predicted_domain = scaling_system.suggest_domain_for_text(test_text)
    print(f"\nПредсказанный домен для '{test_text}': {predicted_domain}")