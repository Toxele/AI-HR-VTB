import requests
from typing import List, Dict, Any
from rag.config import LM_ADDRESS, LM_PORT, MODEL_ID, TEXT_EMBEDDING_MODEL
from document_managament.document_manager import DocumentManager


class RAGSystem:
    def __init__(self):
        self.document_manager = None
        self._is_ready = False
        self.load_data()

    def load_data(self) -> bool:
        self.document_manager = DocumentManager(
            text_embedding_model=TEXT_EMBEDDING_MODEL,
            rebuild_all_indexes=False
        )
        self._is_ready = True
        return True

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

    def generate_candidate_report(self, candidate_info: Dict, job_text: str) -> Dict[str, Any]:
        """Генерирует отчет по кандидату"""
        resume_text = candidate_info['resume_text']

        # Анализ соответствия
        analysis = self.document_manager.analyze_candidate_fit(resume_text, job_text)

        # Генерация детального отчета
        system_prompt = """Ты - HR-специалист. Проанализируй соответствие кандидата требованиям вакансии и создай подробный отчет."""

        user_prompt = f"""
ТРЕБОВАНИЯ ВАКАНСИИ:
{job_text}

РЕЗЮМЕ КАНДИДАТА:
{resume_text}

ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ:
- Общий балл: {analysis['total_score']:.2f}
- Технические навыки: {analysis['technical_skills']['score']:.2f}
- Опыт работы: {analysis['experience']['score']:.2f}
- Образование: {analysis['education']['score']:.2f}

Создай подробный отчет с рекомендацией.
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

    def evaluate_candidates(self, top_n: int = 5) -> List[Dict]:
        """Оценивает кандидатов и возвращает результаты"""
        if not self._is_ready:
            raise RuntimeError("System not ready. Call load_data() first")

        # Сопоставление кандидатов с вакансией
        matches = self.document_manager.match_candidates_to_job(top_n=top_n)

        # Получаем текст требований вакансии
        job_texts = []
        for job_doc in self.document_manager.job_requirement_documents:
            job_texts.extend(job_doc.get_all_text())
        job_combined = " ".join(job_texts)

        results = []
        for score, candidate_info in matches:
            report = self.generate_candidate_report(candidate_info, job_combined)
            results.append({
                'similarity_score': score,
                'report': report,
                'candidate_name': candidate_info['metadata']['document_name']
            })

        return results