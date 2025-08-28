import os
import logging
from rag.core import RAGSystem
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def create_document_folders():
    """Создает необходимые папки для документов"""
    folders = ['documents/pdf/', 'documents/docx/', 'documents/rtf/']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Создана папка: {folder}")


def calculate_metrics(true_labels, predicted_scores, threshold=0.6):
    """Вычисляет метрики качества подбора"""
    predicted_labels = [1 if score >= threshold else 0 for score in predicted_scores]

    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': np.mean([1 if p == t else 0 for p, t in zip(predicted_labels, true_labels)]),
        'threshold': threshold
    }


def main():
    """Основная функция для оценки кандидатов"""
    logger.info("Starting HR Candidate Evaluation System...")

    # Создаем папки для документов
    create_document_folders()

    print("Разместите документы в соответствующих папках:")
    print("- Резюме кандидатов: documents/pdf/, documents/docx/, documents/rtf/")
    print("- Требования вакансий: documents/pdf/, documents/docx/, documents/rtf/")
    print("Названия файлов с ключевыми словами 'resume', 'cv', 'job', 'vacancy' будут автоматически классифицированы")
    print()

    try:
        # Инициализация системы
        rag_system = RAGSystem()
        logger.info("System initialized successfully.")

        # Оценка кандидатов
        results = rag_system.evaluate_candidates(top_n=5)

        if not results:
            print("Не найдено документов для анализа. Разместите файлы в папках documents/")
            return

        print("=" * 80)
        print("HR CANDIDATE EVALUATION REPORT")
        print("=" * 80)
        print(f"Проанализировано кандидатов: {len(results)}")
        print()

        for i, result in enumerate(results, 1):
            print(f"\n{'=' * 50}")
            print(f"CANDIDATE #{i}: {result['candidate_name']}")
            print(f"{'=' * 50}")
            print(f"Similarity Score: {result['similarity_score']:.3f}")
            print(f"Total Fit Score: {result['report']['analysis']['total_score']:.2f}")
            print(f"Recommendation: {result['report']['analysis']['recommendation']}")

            print(f"\nTechnical Skills Match: {result['report']['analysis']['technical_skills']['score']:.2f}")
            print("Matched Skills:", result['report']['analysis']['technical_skills']['matched_skills'])
            print("Missing Skills:", result['report']['analysis']['technical_skills']['missing_skills'])

            print(f"\nDetailed Analysis:")
            print("-" * 50)
            print(result['report']['detailed_report'])
            print("\n")

        # Вычисление метрик (для демонстрации)
        predicted_scores = [result['report']['analysis']['total_score'] for result in results]
        true_labels = [1] * len(results)  # Пример истинных меток

        metrics = calculate_metrics(true_labels, predicted_scores)

        print("\n" + "=" * 80)
        print("SYSTEM PERFORMANCE METRICS")
        print("=" * 80)
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Threshold: {metrics['threshold']}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()