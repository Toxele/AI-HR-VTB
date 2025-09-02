import requests
from typing import List, Dict, Any, Optional
from config import LM_ADDRESS, LM_PORT, MODEL_ID, TEXT_EMBEDDING_MODEL
from document_managament.document_manager import DocumentManager
import os


class RAGSystem:
    def __init__(self):
        self.document_manager = None
        self._is_ready = False
        self.load_data()

    def load_data(self) -> bool:
        self.document_manager = DocumentManager(
            text_embedding_model=TEXT_EMBEDDING_MODEL,
            rebuild_all_indexes=False,
            use_pretrained=True,  # Используем предобученные эмбеддинги
            use_word2vec=False,
            enable_dynamic_scaling=True  # Включаем динамическое масштабирование
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
            'max_tokens': 512,
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
        system_prompt = """Ты - опытный HR-специалист. Проанализируй соответствие кандидата требованиям вакансии и создай подробный отчет с обоснованием рекомендации. Выяви противоречия, если они есть и составь список вопросов, которые ты бы задал на устном собеседовании с кандидатом"""

        user_prompt = f"""
ВАКАНСИЯ: {job_name}

ТРЕБОВАНИЯ ВАКАНСИИ:
{job_text[:2000]}...

РЕЗЮМЕ КАНДИДАТА:
{resume_text[:2000]}...

ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ:
- Общий балл соответствия: {analysis['total_score']:.2f}/1.0
- Навыки: {analysis['technical_skills']['score']:.2f}
- Опыт работы: {analysis['experience']['score']:.2f}
- Образование: {analysis['education']['score']:.2f}
- Языки: {analysis['language_skills']['score']:.2f}

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
        # Используем DocumentManager для анализа
        domain_analysis = self.document_manager._analyze_domain_compatibility(job_text, resume_text)

        # Вычисляем общий score на основе similarity score
        similarity_score = self.document_manager._calculate_vacancy_specific_similarity(
            job_text, resume_text, job_name
        )

        # Создаем структуру анализа
        analysis = {
            'total_score': similarity_score,
            'technical_skills': {
                'score': similarity_score * 0.8,  # Вес технических навыков
                'matched_skills': domain_analysis.get('shared_keywords', []),
                'missing_skills': domain_analysis.get('missing_keywords', [])
            },
            'experience': {
                'score': similarity_score * 0.7  # Вес опыта
            },
            'education': {
                'score': similarity_score * 0.6  # Вес образования
            },
            'language_skills': {
                'score': similarity_score * 0.5  # Вес языковых навыков
            },
            'recommendation': self.document_manager._get_recommendation(similarity_score, domain_analysis),
            'improvement_suggestions': self._generate_improvement_suggestions(domain_analysis)
        }

        return analysis

    def _generate_improvement_suggestions(self, domain_analysis: Dict) -> List[str]:
        """Генерирует предложения по улучшению"""
        suggestions = []

        missing_skills = domain_analysis.get('missing_keywords', [])
        if missing_skills:
            suggestions.append(f"Рекомендуется изучить: {', '.join(missing_skills[:3])}")

        if domain_analysis.get('domain_compatibility', 0) < 0.5:
            suggestions.append("Рассмотреть возможность переквалификации в смежную область")

        return suggestions

    def evaluate_candidates(self, top_n: int = 5) -> List[Dict]:
        """Оценивает кандидатов и возвращает результаты"""
        if not self._is_ready:
            raise RuntimeError("System not ready. Call load_data() first")

        # Сопоставление кандидатов с вакансиями
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

        Args:
            resume_name: Имя файла резюме
            vacancy_name: Имя файла вакансии

        Returns:
            Словарь с информацией о соответствии
        """
        if not self._is_ready:
            raise RuntimeError("System not ready. Call load_data() first")

        return self.document_manager.get_specific_match(resume_name, vacancy_name)