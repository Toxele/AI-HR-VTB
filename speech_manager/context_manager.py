from scipy.signal import hilbert2

from rag.core import RAGSystem


class ContextManager:

    def __init__(self, resume_filename, vacancy_filename):
        self.rag_system = RAGSystem()
        self.history = [
            {
                "role": "system",
                "content": "АНАЛИЗ РЕЗЮМЕ", # можно будет местами поменять metadata и content
                "metadata": self.rag_system.get_candidate_evaluation(resume_filename, vacancy_filename)
            }
        ]

    def query(self, user_query):
        self.history.append(
            {
                "role": "user",
                "content": f"{user_query}",
            }
        )
        return self.rag_system.generate_answer_with_history(history=self.history)

