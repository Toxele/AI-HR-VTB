import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from typing import Dict, List, Tuple
from .document_factory import DocumentFactory


class DocumentManager:
    def __init__(self,
                 text_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 rebuild_all_indexes=False
                 ):
        self.text_embedding_model = text_embedding_model
        self.text_embeddings = HuggingFaceEmbeddings(model_name=self.text_embedding_model)
        self.base_documents_path = '../documents/'
        self.resume_documents = []
        self.job_requirement_documents = []
        self.rebuild_all_indexes = rebuild_all_indexes
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.load_documents()

    def load_documents(self):
        """Загружает все документы из соответствующих папок"""
        document_folders = {
            'pdf': 'pdf/',
            'docx': 'docx/',
            'rtf': 'rtf/'
        }

        for file_type, folder_name in document_folders.items():
            folder_path = os.path.join(self.base_documents_path, folder_name)
            if os.path.exists(folder_path):
                files_info = self.get_files_info(folder_path, f'.{file_type}')

                for file_info in files_info:
                    document = DocumentFactory.create_document(
                        document_name=file_info['document_name'],
                        document_dir=folder_path,
                        extract_dir_name=file_info['extract_dir_name'],
                        text_embeddings=self.text_embeddings,
                        rebuild_index=self.rebuild_all_indexes
                    )

                    if document:
                        doc_type = self._classify_document_type(file_info['document_name'])
                        if doc_type == 'resume':
                            self.resume_documents.append(document)
                        else:
                            self.job_requirement_documents.append(document)

    def _classify_document_type(self, filename: str) -> str:
        """Классифицирует документ как резюме или требования к вакансии"""
        resume_keywords = ['resume', 'cv', 'кандидат', 'соискатель', 'резюме', 'candidate']
        job_keywords = ['job', 'vacancy', 'вакансия', 'требования', 'position', 'jd', 'description']

        filename_lower = filename.lower()

        if any(keyword in filename_lower for keyword in resume_keywords):
            return 'resume'
        elif any(keyword in filename_lower for keyword in job_keywords):
            return 'job_requirements'
        else:
            # Анализ содержимого для классификации
            return self._classify_by_content(filename)

    def _classify_by_content(self, filename: str) -> str:
        """Классифицирует документ по содержимому"""
        # Временная реализация - можно улучшить с помощью ML
        return 'resume'  # По умолчанию считаем резюме

    def match_candidates_to_job(self, top_n: int = 5) -> List[Tuple[float, Dict]]:
        """Сопоставляет кандидатов с требованиями вакансии"""
        if not self.job_requirement_documents or not self.resume_documents:
            print("Нет документов для анализа")
            return []

        # Получаем текст требований вакансии
        job_texts = []
        for job_doc in self.job_requirement_documents:
            job_texts.extend(job_doc.get_all_text())

        job_combined = " ".join(job_texts)

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
            print("Не удалось извлечь текст из резюме")
            return []

        # Вычисляем TF-IDF схожесть
        all_texts = [job_combined] + resume_texts
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)

        # Сравниваем каждое резюме с требованиями
        job_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(job_vector, resume_vectors).flatten()

        # Сортируем по убыванию схожести
        results = []
        for i, similarity in enumerate(similarities):
            results.append((similarity, {
                'similarity_score': float(similarity),
                'resume_text': resume_texts[i],
                'metadata': resume_metadata[i]
            }))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_n]

    def analyze_candidate_fit(self, resume_text: str, job_text: str) -> Dict:
        """Анализирует соответствие кандидата требованиям вакансии"""
        technical_skills = self._extract_skills(resume_text, job_text)
        experience_match = self._analyze_experience(resume_text, job_text)
        education_match = self._analyze_education(resume_text, job_text)

        total_score = (
                technical_skills['score'] * 0.5 +
                experience_match['score'] * 0.3 +
                education_match['score'] * 0.2
        )

        return {
            'total_score': total_score,
            'technical_skills': technical_skills,
            'experience': experience_match,
            'education': education_match,
            'recommendation': self._generate_recommendation(total_score),
            'strengths': technical_skills['matched_skills'],
            'weaknesses': technical_skills['missing_skills']
        }

    def _extract_skills(self, resume_text: str, job_text: str) -> Dict:
        """Извлекает и сопоставляет навыки"""
        common_tech_skills = ['python', 'java', 'sql', 'javascript', 'html', 'css',
                              'react', 'angular', 'vue', 'docker', 'kubernetes', 'aws',
                              'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'typescript',
                              'node.js', 'django', 'flask', 'spring', 'laravel']

        resume_skills = []
        job_skills = []

        for skill in common_tech_skills:
            if skill in resume_text.lower():
                resume_skills.append(skill)
            if skill in job_text.lower():
                job_skills.append(skill)

        matched_skills = set(resume_skills) & set(job_skills)
        missing_skills = set(job_skills) - set(resume_skills)

        score = len(matched_skills) / len(job_skills) if job_skills else 0

        return {
            'score': score,
            'matched_skills': list(matched_skills),
            'missing_skills': list(missing_skills),
            'resume_skills': resume_skills,
            'job_required_skills': job_skills
        }

    def _analyze_experience(self, resume_text: str, job_text: str) -> Dict:
        """Анализирует соответствие опыта работы"""
        experience_keywords = ['опыт', 'experience', 'стаж', 'years', 'работал', 'worked', 'experience']
        years_keywords = ['год', 'years', 'лет']

        resume_exp = any(keyword in resume_text.lower() for keyword in experience_keywords)
        job_exp = any(keyword in job_text.lower() for keyword in experience_keywords)

        # Простой анализ лет опыта
        has_years = any(keyword in resume_text.lower() for keyword in years_keywords)

        score = 1.0 if resume_exp and job_exp and has_years else 0.7 if resume_exp else 0.3

        return {'score': score, 'has_experience': resume_exp, 'experience_required': job_exp}

    def _analyze_education(self, resume_text: str, job_text: str) -> Dict:
        """Анализирует соответствие образования"""
        education_keywords = ['образование', 'education', 'университет', 'degree', 'диплом', 'education']
        higher_education = ['бакалавр', 'магистр', 'специалист', 'bachelor', 'master', 'phd']

        resume_edu = any(keyword in resume_text.lower() for keyword in education_keywords)
        job_edu = any(keyword in job_text.lower() for keyword in education_keywords)

        has_higher_edu = any(edu in resume_text.lower() for edu in higher_education)

        score = 1.0 if resume_edu and job_edu and has_higher_edu else 0.7 if resume_edu else 0.3

        return {'score': score, 'has_education': resume_edu, 'education_required': job_edu}

    def _generate_recommendation(self, score: float) -> str:
        """Генерирует рекомендацию на основе оценки"""
        if score >= 0.8:
            return "Рекомендован к следующему этапу"
        elif score >= 0.6:
            return "Требуется дополнительное собеседование"
        else:
            return "Не соответствует требованиям"

    def get_files_info(self, folder_path: str, extension: str) -> List[Dict]:
        """Возвращает список файлов с указанным расширением"""
        files_info = []
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не существует!")
            return files_info

        for file_path in Path(folder_path).glob(f'*{extension}'):
            files_info.append({
                'document_name': file_path.name,
                'extract_dir_name': file_path.stem
            })

        return files_info


# Пример использования
if __name__ == "__main__":
    document_manager = DocumentManager(rebuild_all_indexes=False)
    matches = document_manager.match_candidates_to_job(top_n=3)

    for score, candidate_info in matches:
        print(f"Схожесть: {score:.3f}")
        print(f"Кандидат: {candidate_info['metadata']['document_name']}")
        print("---")