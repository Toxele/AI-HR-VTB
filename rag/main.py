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


def create_folder_structure():
    """Создает структуру папок для документов"""
    base_folders = ['cv', 'vacancy']
    sub_folders = ['pdf', 'docx', 'rtf']

    for base in base_folders:
        for sub in sub_folders:
            folder_path = os.path.join('documents', base, sub)
            os.makedirs(folder_path, exist_ok=True)
            print(f"Создана папка: {folder_path}")

    print("\nРазместите документы в соответствующих папках:")
    print("Резюме кандидатов: documents/cv/pdf/, documents/cv/docx/, documents/cv/rtf/")
    print("Описания вакансий: documents/vacancy/pdf/, documents/vacancy/docx/, documents/vacancy/rtf/")


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

    # Создаем структуру папок
    create_folder_structure()
    print()

    try:
        # Инициализация системы
        rag_system = RAGSystem()
        logger.info("System initialized successfully.")

        # Оценка кандидатов
        results = rag_system.evaluate_candidates(top_n=5)

        if not results:
            print("Не найдено документов для анализа.")
            print("Пожалуйста, разместите файлы в соответствующих папках:")
            print("- Резюме: documents/cv/pdf/, documents/cv/docx/, documents/cv/rtf/")
            print("- Вакансии: documents/vacancy/pdf/, documents/vacancy/docx/, documents/vacancy/rtf/")
            return

        print("=" * 100)
        print("HR CANDIDATE EVALUATION REPORT")
        print("=" * 100)
        print(f"Проанализировано кандидатов: {len(results)}")
        print()

        for i, result in enumerate(results, 1):
            print(f"\n{'=' * 80}")
            print(f"CANDIDATE #{i}: {result['candidate_name']}")
            print(f"FOR JOB: {result['job_name']}")
            print(f"{'=' * 80}")
            print(f"Similarity Score: {result['similarity_score']:.3f}")
            print(f"Total Fit Score: {result['report']['analysis']['total_score']:.2f}/1.0")
            print(f"Recommendation: {result['report']['analysis']['recommendation']}")

            print(f"\nDETAILED ANALYSIS:")
            print(f"Technical Skills: {result['report']['analysis']['technical_skills']['score']:.2f}")
            print(f"Experience: {result['report']['analysis']['experience']['score']:.2f}")
            print(f"Education: {result['report']['analysis']['education']['score']:.2f}")
            print(f"Languages: {result['report']['analysis']['language_skills']['score']:.2f}")

            print(
                f"\nMatched Skills: {', '.join(result['report']['analysis']['technical_skills']['matched_skills'][:10])}")
            if result['report']['analysis']['technical_skills']['missing_skills']:
                print(
                    f"Missing Skills: {', '.join(result['report']['analysis']['technical_skills']['missing_skills'][:5])}")

            if result['report']['analysis']['improvement_suggestions']:
                print(f"\nImprovement Suggestions:")
                for suggestion in result['report']['analysis']['improvement_suggestions']:
                    print(f"  - {suggestion}")

            print(f"\nDetailed Report:")
            print("-" * 80)
            print(result['report']['detailed_report'])
            print("\n")

        # Вычисление метрик
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

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()