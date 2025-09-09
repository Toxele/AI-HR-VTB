import asyncio
from functools import partial
from mistralai import Mistral
import random
from concurrent.futures import ThreadPoolExecutor
from config import API_KEYS, MODEL, MAX_CONCURRENCY


class ServiceTierCapacityExceeded(Exception):
    pass



class ContextManager:

    def __init__(self, resume_text, vacancy_filename):
        self.clients = [Mistral(api_key=k) for k in API_KEYS]
        self.executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY * 2)
        self.history = [
            {
                "role": "system",
                "content": "АНАЛИЗ РЕЗЮМЕ", # можно будет местами поменять metadata и content
                "metadata": self.call_mistral_sync(resume_text  )
            }
        ]

    def query(self, user_query):
        self.history.append(
            {
                "role": "user",
                "content": f"{user_query}",
            }
        )
        # TODO: заменить на свой кастомный запрос
        pass
        #return self.rag_system.generate_answer_with_history(history=self.history)

    def generate_skill_prompt(self, job_text: str) -> str:
        return f"""
        Ты SkillExtractorAI — точный генератор JSON, который извлекает и классифицирует требуемые навыки из описаний вакансий.

        Правила:
        1. Извлекай ВСЕ навыки (hard skills - технические, soft skills - межличностные)
        2. Приводи к стандартной форме
        3. Не объединяй разные навыки

        Формат вывода:
        [{{"s": "навык", "t": "H|S"}}, ...]

        Вакансия:
        {job_text}
        """

    def call_mistral_sync(self, job_text: str) -> str | None:
        try:
            client = random.choice(self.clients)
            response = client.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": self.generate_skill_prompt(job_text)}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                return None

            # Проверка на ошибки API
            if hasattr(response, 'errors') and response.errors:
                error = response.errors[0].get('code')
                if '3505' in error.lower():
                    raise ServiceTierCapacityExceeded(error)

            return content

        except Exception as e:
            error_msg = str(e)
            if '3505' in error_msg.lower():
                raise ServiceTierCapacityExceeded(error_msg)
            print(error_msg)
            return None

    async def call_mistral(self, job_text: str) -> str | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, partial(self.call_mistral_sync, job_text))

