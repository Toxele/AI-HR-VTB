import requests
from typing import List, Dict, Any, Optional, Set, Tuple
from config import LM_ADDRESS, LM_PORT, MODEL_ID, TEXT_EMBEDDING_MODEL
from document_managament.document_manager import DocumentManager
import os
import re
import numpy as np


class RAGSystem:
    def __init__(self):
        self.document_manager = None
        self._is_ready = False
        self.load_data()

    def load_data(self) -> bool:
        self.document_manager = DocumentManager(
            text_embedding_model=TEXT_EMBEDDING_MODEL,
            rebuild_all_indexes=False,
            use_pretrained=True,
            use_word2vec=False,
            enable_dynamic_scaling=True
        )
        self._is_ready = True
        return True

    def get_documents_info(self) -> Dict:
        """Возвращает информацию о загруженных документах"""
        if not self._is_ready:
            return {'resumes_count': 0, 'vacancies_count': 0, 'embedding_mode': 'not_ready'}

        return self.document_manager.get_documents_info()

    @staticmethod
    def query_llm(messages: list, **kwargs) -> Dict[str, str]:
        """Запрос к языковой модели"""
        headers = {'Content-Type': 'application/json'}
        params = {
            'model': MODEL_ID,
            'messages': messages,
            'temperature': 0.2,
            'max_tokens': 1024,  # Увеличили для более детальных отчетов
        }

        try:
            response = requests.post(
                f'{LM_ADDRESS}:{LM_PORT}/v1/chat/completions',
                headers=headers,
                json=params,
                timeout=540
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']
        except Exception as e:
            return {'role': 'error', 'content': f'Error: {str(e)}'}

    def generate_candidate_report(self, candidate_info: Dict) -> Dict[str, Any]:
        """Генерирует отчет по кандидату"""
        resume_text = candidate_info['resume_text']
        job_text = candidate_info['job_text']
        job_name = candidate_info['job_name']

        # Анализ соответствия
        analysis = self.analyze_candidate_fit(resume_text, job_text, job_name)

        # Генерация детального отчета
        system_prompt = """Ты - опытный HR-специалист. Проанализируй соответствие кандидата требованиям вакансии и создай подробный отчет с обоснованием рекомендации. Выяви противоречия, если они есть и составь список вопросов, которые ты бы задал на устном собеседовании с кандидатом."""

        user_prompt = f"""
ВАКАНСИЯ: {job_name}

ТРЕБОВАНИЯ ВАКАНСИИ:
{job_text[:2000]}...

РЕЗЮМЕ КАНДИДАТА:
{resume_text[:2000]}...

ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ:
- Общий балл соответствия: {analysis['total_score']:.2f}/1.0
- Технические навыки: {analysis['technical_skills']['score']:.2f} ({analysis['technical_skills']['matched_count']}/{analysis['technical_skills']['total_required']})
- Опыт работы: {analysis['experience']['score']:.2f} ({analysis['experience']['total_years']} лет из {analysis['experience']['required_years']})
- Образование: {analysis['education']['score']:.2f} ({analysis['education']['highest_level']} vs {analysis['education']['required_level']})
- Языки: {analysis['language_skills']['score']:.2f} ({len(analysis['language_skills']['matched_languages'])}/{len(analysis['language_skills']['required_languages'])})

СОВПАДАЮЩИЕ НАВЫКИ: {', '.join(analysis['technical_skills']['matched_skills'][:5])}
ОТСУТСТВУЮЩИЕ НАВЫКИ: {', '.join(analysis['technical_skills']['missing_skills'][:3])}

Создай подробный профессиональный отчет с рекомендацией для HR-отдела.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        llm_response = self.query_llm(messages)

        return {
            'analysis': analysis,
            'detailed_report': llm_response['content'],
            'candidate_info': candidate_info
        }

    def analyze_candidate_fit(self, resume_text: str, job_text: str, job_name: str) -> Dict[str, Any]:
        """Анализирует соответствие кандидата требованиям вакансии"""
        # Индивидуальный анализ для каждой категории
        technical_analysis = self._analyze_technical_skills(resume_text, job_text)
        experience_analysis = self._analyze_experience(resume_text, job_text)
        education_analysis = self._analyze_education(resume_text, job_text)
        language_analysis = self._analyze_languages(resume_text, job_text)

        # Взвешенное общее score
        weights = {
            'technical_skills': 0.4,
            'experience': 0.3,
            'education': 0.2,
            'language_skills': 0.1
        }

        total_score = (
                technical_analysis['score'] * weights['technical_skills'] +
                experience_analysis['score'] * weights['experience'] +
                education_analysis['score'] * weights['education'] +
                language_analysis['score'] * weights['language_skills']
        )

        # Генерация рекомендации
        recommendation = self._generate_recommendation({
            'technical': technical_analysis,
            'experience': experience_analysis,
            'education': education_analysis,
            'languages': language_analysis
        })

        return {
            'total_score': total_score,
            'technical_skills': technical_analysis,
            'experience': experience_analysis,
            'education': education_analysis,
            'language_skills': language_analysis,
            'recommendation': recommendation,
            'improvement_suggestions': self._generate_improvement_suggestions({
                'technical': technical_analysis,
                'experience': experience_analysis,
                'education': education_analysis,
                'languages': language_analysis
            })
        }

    def _analyze_technical_skills(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """Детальный анализ технических навыков"""
        resume_skills = self._extract_technical_skills(resume_text)  # Только 1 аргумент
        job_skills = self._extract_technical_skills(job_text)  # Только 1 аргумент

        matched_skills = resume_skills & job_skills
        missing_skills = job_skills - resume_skills
        extra_skills = resume_skills - job_skills

        if not job_skills:
            score = 1.0  # Если нет требований - идеальное соответствие
        else:
            score = len(matched_skills) / len(job_skills)

        return {
            'score': min(1.0, score),
            'matched_skills': list(matched_skills),
            'missing_skills': list(missing_skills),
            'extra_skills': list(extra_skills),
            'total_required': len(job_skills),
            'matched_count': len(matched_skills)
        }

    def _extract_technical_skills(self, text: str) -> Set[str]:
        """Извлечение технических навыков из текста"""
        tech_keywords = {
            'python', 'java', 'javascript', 'typescript', 'sql', 'nosql', 'html', 'css',
            'react', 'angular', 'vue', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'django', 'flask', 'spring', 'hibernate', 'node', 'express', 'laravel',
            'postgresql', 'mysql', 'mongodb', 'redis', 'nginx', 'apache', 'linux',
            'git', 'jenkins', 'ci/cd', 'devops', 'ml', 'ai', 'tensorflow', 'pytorch'
        }

        words = set(re.findall(r'\b[a-zа-я]+\b', text.lower()))
        return words & tech_keywords

    def _analyze_experience(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """Анализ опыта работы"""
        resume_years = self._extract_experience_years(resume_text)
        job_required_years = self._extract_required_experience(job_text)

        if job_required_years == 0:
            score = 1.0  # Если опыт не требуется
        elif resume_years >= job_required_years:
            score = 1.0  # Опыт достаточный
        else:
            score = resume_years / job_required_years  # Частичное соответствие

        return {
            'score': min(1.0, score),
            'total_years': resume_years,
            'required_years': job_required_years
        }

    def _extract_experience_years(self, text: str) -> int:
        """Извлечение лет опыта из резюме"""
        patterns = [
            r'опыт.*?(\d+).*?год',
            r'стаж.*?(\d+).*?год',
            r'experience.*?(\d+).*?year',
            r'(\d+).*?лет.*?опыт',
            r'(\d+).*?years.*?experience'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return 0

    def _extract_required_experience(self, text: str) -> int:
        """Извлечение требуемого опыта из вакансии"""
        patterns = [
            r'требуется.*?опыт.*?(\d+).*?год',
            r'опыт.*?работы.*?(\d+).*?год',
            r'experience.*?(\d+).*?years',
            r'(\d+).*?лет.*?опыт.*?требуется'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return 0

    def _analyze_education(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """Анализ образования с исправленной логикой сравнения"""
        resume_education = self._extract_education_level(resume_text)
        job_education = self._extract_required_education(job_text)

        # Правильная иерархия уровней образования с числовыми значениями
        education_levels = {
            'школа': 1, 'среднее': 2, 'колледж': 3, 'техникум': 3,
            'бакалавр': 4, 'специалист': 5, 'магистр': 6,
            'кандидат': 7, 'доктор': 8, 'phd': 7,
            'высшее': 4  # базовый уровень высшего образования
        }

        # Получаем числовые значения уровней
        resume_level = education_levels.get(resume_education.lower() if resume_education else '', 0)
        job_level = education_levels.get(job_education.lower() if job_education else '', 0)

        # Если образование не требуется или не указано
        if job_level == 0:
            score = 1.0  # Если образование не требуется
        elif resume_level == 0:
            score = 0.0  # Если у кандидата не указано образование, но оно требуется
        elif resume_level >= job_level:
            score = 1.0  # Образование достаточное или выше требуемого
        else:
            # Частичное соответствие - линейная шкала от 0 до 1
            score = resume_level / job_level

        return {
            'score': min(1.0, max(0.0, score)),  # Обеспечиваем диапазон 0-1
            'highest_level': resume_education,
            'required_level': job_education,
            'resume_level_value': resume_level,
            'required_level_value': job_level
        }

    def _extract_education_level(self, text: str) -> str:
        """Извлечение уровня образования из текста"""
        text_lower = text.lower()

        # Приоритетный поиск по уровням (от высшего к низшему)
        levels_priority = [
            ('доктор', 'докторская', 'doctor', 'phd'),
            ('кандидат', 'кандидатская', 'candidate'),
            ('магистр', 'магистратура', 'master'),
            ('специалист', 'специалитет', 'specialist'),
            ('бакалавр', 'бакалавриат', 'bachelor'),
            ('высшее', 'высшее образование', 'higher education'),
            ('колледж', 'техникум', 'college', 'technical'),
            ('среднее', 'среднее образование', 'secondary'),
            ('школа', 'школьное', 'school')
        ]

        for level_group in levels_priority:
            for level_keyword in level_group:
                if level_keyword in text_lower:
                    return level_group[0]  # Возвращаем основной уровень из группы

        return ''  # Если уровень не найден

    def _extract_required_education(self, text: str) -> str:
        """Извлечение требуемого образования из текста вакансии"""
        text_lower = text.lower()

        requirements = [
            (['высшее', 'higher', 'образование'], 'высшее'),
            (['бакалавр', 'bachelor'], 'бакалавр'),
            (['магистр', 'master'], 'магистр'),
            (['специалист', 'specialist'], 'специалист'),
            (['кандидат', 'phd', 'аспирант'], 'кандидат'),
            (['доктор', 'doctor'], 'доктор'),
            (['среднее', 'secondary'], 'среднее'),
            (['школа', 'school'], 'школа')
        ]

        for keywords, level in requirements:
            if any(keyword in text_lower for keyword in keywords):
                return level

        return ''  # Если требования не указаны

    def _analyze_languages(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """Анализ языковых навыков"""
        resume_languages = self._extract_languages(resume_text)
        job_languages = self._extract_required_languages(job_text)

        if not job_languages:
            return {
                'score': 1.0,
                'matched_languages': [],
                'required_languages': {},
                'resume_languages': resume_languages
            }

        matched_count = 0
        matched_languages = []

        for lang, required_level in job_languages.items():
            if lang in resume_languages:
                resume_level = resume_languages[lang]
                if self._compare_language_levels(resume_level, required_level):
                    matched_count += 1
                    matched_languages.append(f"{lang} ({resume_level})")

        score = matched_count / len(job_languages) if job_languages else 1.0

        return {
            'score': min(1.0, score),
            'matched_languages': matched_languages,
            'required_languages': job_languages,
            'resume_languages': resume_languages
        }

    def _extract_languages(self, text: str) -> Dict[str, str]:
        """Извлечение языков из текста"""
        languages = {}
        patterns = [
            r'(английский|english).*?(базовый|средний|продвинутый|носитель|basic|intermediate|advanced|native)',
            r'(немецкий|german).*?(базовый|средний|продвинутый|носитель)',
            r'(французский|french).*?(базовый|средний|продвинутый|носитель)',
            r'(китайский|chinese).*?(базовый|средний|продвинутый|носитель)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for lang, level in matches:
                languages[lang] = level

        return languages

    def _extract_required_languages(self, text: str) -> Dict[str, str]:
        """Извлечение требуемых языков"""
        required_languages = {}
        patterns = [
            r'(английский|english).*?(требуется|необходим|обязателен).*?(базовый|средний|продвинутый|носитель)',
            r'(немецкий|german).*?(требуется|необходим|обязателен)',
            r'(французский|french).*?(требуется|необходим|обязателен)',
            r'знание.*?(английск|немецк|французск|китайск).*?язык'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    lang = match[0]
                    level = match[2] if len(match) > 2 else 'средний'
                else:
                    lang = match
                    level = 'средний'
                required_languages[lang] = level

        return required_languages

    def _compare_language_levels(self, resume_level: str, required_level: str) -> bool:
        """Сравнение уровней владения языком"""
        level_order = {
            'базовый': 1, 'basic': 1,
            'средний': 2, 'intermediate': 2,
            'продвинутый': 3, 'advanced': 3,
            'носитель': 4, 'native': 4
        }

        resume_score = level_order.get(resume_level.lower(), 0)
        required_score = level_order.get(required_level.lower(), 0)

        return resume_score >= required_score

    def _generate_recommendation(self, analysis: Dict[str, Any]) -> str:
        """Генерация рекомендации на основе анализа"""
        tech_score = analysis['technical']['score']
        exp_score = analysis['experience']['score']
        edu_score = analysis['education']['score']
        lang_score = analysis['languages']['score']

        if tech_score >= 0.8 and exp_score >= 0.7:
            return "Сильное соответствие. Рекомендуем к собеседованию."
        elif tech_score >= 0.7:
            return "Хорошее техническое соответствие. Требуется оценка soft skills."
        elif exp_score >= 0.8:
            return "Релевантный опыт. Требуется оценка технических навыков."
        elif tech_score >= 0.6 and exp_score >= 0.6:
            return "Умеренное соответствие. Рассмотреть после дополнительного собеседования."
        else:
            return "Требуется дополнительная оценка. Рассмотреть других кандидатов."

    def _generate_improvement_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Генерация предложений по улучшению"""
        suggestions = []

        # Технические навыки
        if analysis['technical']['score'] < 0.7 and analysis['technical']['missing_skills']:
            missing = analysis['technical']['missing_skills'][:3]
            suggestions.append(f"Изучить недостающие навыки: {', '.join(missing)}")

        # Опыт работы
        if analysis['experience']['score'] < 0.7:
            diff = analysis['experience']['required_years'] - analysis['experience']['total_years']
            if diff > 0:
                suggestions.append(f"Накопить еще {diff} лет релевантного опыта")

        # Образование
        if analysis['education']['score'] < 0.8:
            suggestions.append("Рассмотреть возможность получения дополнительного образования")

        # Языки
        if analysis['languages']['score'] < 1.0:
            suggestions.append("Улучшить знание требуемых иностранных языков")

        return suggestions

    def evaluate_candidates(self, top_n: int = 5) -> List[Dict]:
        """Оценивает кандидатов и возвращает результаты"""
        if not self._is_ready:
            raise RuntimeError("System not ready. Call load_data() first")

        # Используем публичный метод для сопоставления
        matches = self.document_manager.match_candidates_to_job(top_n=top_n)

        results = []
        for score, candidate_info in matches:
            report = self.generate_candidate_report(candidate_info)
            results.append({
                'similarity_score': score,
                'report': report,
                'candidate_name': candidate_info['metadata']['document_name'],
                'job_name': candidate_info['job_name']
            })

        return results

    def get_specific_match(self, resume_name: str, vacancy_name: str) -> Optional[Dict[str, Any]]:
        """
        Находит соответствие конкретного резюме конкретной вакансии
        """
        if not self._is_ready:
            raise RuntimeError("System not ready. Call load_data() first")

        # Используем публичный метод DocumentManager
        return self.document_manager.get_specific_match(resume_name, vacancy_name)