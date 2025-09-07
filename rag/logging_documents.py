import os
from typing import Dict
from rag.core import RAGSystem


def evaluate_single_candidate(resume_filename: str, vacancy_filename: str) -> str:
    """
    Оценивает одного кандидата для одной вакансии и возвращает строку с отчетом
    """
    # Инициализация системы
    rag_system = RAGSystem()

    # Используем новую функцию, которая работает с уже загруженными документами
    return rag_system.get_candidate_evaluation(resume_filename, vacancy_filename)



# Пример использования
if __name__ == "__main__":
    resume_file = "Образец резюме 1 Бизнес аналитик.rtf"
    vacancy_file = "Описание бизнес аналитик.docx"

    report = evaluate_single_candidate(resume_file, vacancy_file)
    print(report)