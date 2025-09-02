import os
import logging
from typing import Dict, List, Any
from rag.core import RAGSystem
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from datetime import datetime
from collections import Counter

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f"hr_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_separator(title: str = "", width: int = 100, char: str = "="):
    """Печать разделителя с заголовком"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


def print_colored(text: str, color: str = "white", style: str = "normal"):
    """Печать цветного текста"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }

    styles = {
        "normal": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }

    color_code = colors.get(color, "\033[97m")
    style_code = styles.get(style, "\033[0m")
    reset_code = "\033[0m"

    print(f"{style_code}{color_code}{text}{reset_code}")


def create_folder_structure():
    """Создает структуру папок для документов"""
    print_separator("📁 СТРУКТУРА ПАПОК ДОКУМЕНТОВ", 80)

    base_folders = ['cv', 'vacancy']
    sub_folders = ['pdf', 'docx', 'rtf']

    for base in base_folders:
        for sub in sub_folders:
            folder_path = os.path.join('documents', base, sub)
            os.makedirs(folder_path, exist_ok=True)
            print_colored(f"📂 Создана папка: {folder_path}", "green")

    print_colored("\n📍 Разместите документы в соответствующих папках:", "yellow", "bold")
    print_colored("   📄 Резюме кандидатов: documents/cv/pdf/, documents/cv/docx/, documents/cv/rtf/", "cyan")
    print_colored("   📋 Описания вакансий: documents/vacancy/pdf/, documents/vacancy/docx/, documents/vacancy/rtf/",
                  "cyan")


def print_system_info(docs_info: Dict):
    """Вывод информации о системе и документах"""
    print_separator("🏢 ИНФОРМАЦИЯ О СИСТЕМЕ", 100)

    print_colored("📊 ЗАГРУЖЕННЫЕ ДОКУМЕНТЫ:", "blue", "bold")
    print_colored(f"   📄 Резюме кандидатов: {docs_info['resumes_count']}",
                  "green" if docs_info['resumes_count'] > 0 else "red")
    print_colored(f"   📋 Вакансий: {docs_info['vacancies_count']}",
                  "green" if docs_info['vacancies_count'] > 0 else "red")

    print_colored("\n🔧 НАСТРОЙКИ СИСТЕМЫ:", "blue", "bold")
    print_colored(f"   🎯 Режим эмбеддингов: {docs_info['embedding_mode'].upper()}", "cyan")
    print_colored(
        f"   ⚡ Динамическое масштабирование: {'✅ ВКЛЮЧЕНО' if docs_info['dynamic_scaling_enabled'] else '❌ ВЫКЛЮЧЕНО'}",
        "green" if docs_info['dynamic_scaling_enabled'] else "yellow")

    # Информация о моделях
    if docs_info.get('pretrained_model_loaded', False):
        print_colored(f"   🤖 Предобученная модель: ✅ ЗАГРУЖЕНА (словарь: {docs_info.get('vocabulary_size', 'N/A')})",
                      "green")
    elif docs_info.get('word2vec_usable', False):
        print_colored(f"   🤖 Word2Vec модель: ✅ ГОТОВА (словарь: {docs_info.get('word2vec_vocabulary_size', 'N/A')})",
                      "green")
    else:
        print_colored("   🤖 Модели: ⚠️  ОГРАНИЧЕННЫЕ ВОЗМОЖНОСТИ", "yellow")

    # Информация о доменах
    if docs_info.get('domains_count', 0) > 0:
        print_colored(f"   🌐 Домены: {docs_info['domains_count']} обнаружено", "magenta")
        print_colored(f"   🎯 Поддомены: {docs_info['subdomains_count']} обнаружено", "magenta")


def print_documents_list(rag_system: RAGSystem):
    """Вывод списка загруженных документов"""
    print_separator("📋 СПИСОК ДОКУМЕНТОВ", 100)

    # Получаем информацию о документах через document_manager
    doc_manager = rag_system.document_manager

    print_colored("📄 РЕЗЮМЕ КАНДИДАТОВ:", "blue", "bold")
    if doc_manager.resume_documents:
        for i, doc in enumerate(doc_manager.resume_documents, 1):
            file_type = os.path.splitext(doc.document_name)[1].upper()
            print_colored(f"   {i:2d}. {doc.document_name} ({file_type})", "cyan")
    else:
        print_colored("   ❌ Нет загруженных резюме", "red")

    print_colored("\n📋 ОПИСАНИЯ ВАКАНСИЙ:", "blue", "bold")
    if doc_manager.job_requirement_documents:
        for i, doc in enumerate(doc_manager.job_requirement_documents, 1):
            file_type = os.path.splitext(doc.document_name)[1].upper()
            print_colored(f"   {i:2d}. {doc.document_name} ({file_type})", "cyan")
    else:
        print_colored("   ❌ Нет загруженных вакансий", "red")


def print_domain_analysis(rag_system: RAGSystem):
    """Вывод анализа доменов"""
    if not rag_system.document_manager.enable_dynamic_scaling:
        return

    domain_analysis = rag_system.document_manager.get_domain_analysis()
    if not domain_analysis:
        return

    print_separator("🌐 АНАЛИЗ ДОМЕНОВ И ПОДДОМЕНОВ", 100)

    domains = domain_analysis.get('domains', {})
    subdomains = domain_analysis.get('subdomains', {})
    weights = domain_analysis.get('domain_weights', {})

    if domains:
        print_colored("🏢 ОСНОВНЫЕ ДОМЕНЫ:", "blue", "bold")
        for domain, count in list(domains.items())[:10]:  # Показываем топ-10
            weight = weights.get(domain, 1.0)
            print_colored(f"   • {domain}: {count} упоминаний (вес: {weight:.2f})", "magenta")

    if subdomains:
        print_colored("\n🎯 ПОДДОМЕНЫ:", "blue", "bold")
        for subdomain, count in list(subdomains.items())[:15]:  # Показываем топ-15
            print_colored(f"   • {subdomain}: {count} упоминаний", "cyan")

    if len(domains) > 10:
        print_colored(f"\n   ... и еще {len(domains) - 10} доменов", "white")
    if len(subdomains) > 15:
        print_colored(f"   ... и еще {len(subdomains) - 15} поддоменов", "white")


def print_matching_process_info(rag_system: RAGSystem):
    """Вывод информации о процессе сопоставления"""
    print_separator("🔍 ПРОЦЕСС СОПОСТАВЛЕНИЯ", 100)

    doc_manager = rag_system.document_manager
    resumes_count = len(doc_manager.resume_documents)
    vacancies_count = len(doc_manager.job_requirement_documents)

    print_colored(f"📊 БУДЕТ ПРОАНАЛИЗИРОВАНО:", "blue", "bold")
    print_colored(f"   👥 Кандидатов: {resumes_count}", "cyan")
    print_colored(f"   💼 Вакансий: {vacancies_count}", "cyan")
    print_colored(f"   🔄 Всего комбинаций: {resumes_count * vacancies_count}", "cyan")

    print_colored(f"\n🎯 РЕЖИМ АНАЛИЗА:", "blue", "bold")
    if doc_manager.use_pretrained:
        print_colored("   🤖 Используются предобученные эмбеддинги", "green")
    elif doc_manager.use_word2vec:
        print_colored("   📚 Используется Word2Vec с обучением на данных", "green")
    else:
        print_colored("   📊 Используется TF-IDF анализ", "green")


def print_candidate_report(result: Dict, candidate_num: int):
    """Детальный вывод отчета по кандидату"""
    similarity_score = result['similarity_score']
    total_score = result['report']['analysis']['total_score']

    # Выбор цвета в зависимости от оценки
    if similarity_score >= 0.8:
        color = "green"
        emoji = "🥇"
    elif similarity_score >= 0.6:
        color = "cyan"
        emoji = "🥈"
    elif similarity_score >= 0.4:
        color = "yellow"
        emoji = "🥉"
    else:
        color = "red"
        emoji = "⚠️ "

    print_separator(f"{emoji} КАНДИДАТ #{candidate_num}: {result['candidate_name']}", 100, "=")

    print_colored(f"🎯 Вакансия: {result['job_name']}", "blue", "bold")
    print_colored(f"📊 Семантическая схожесть с вакансией: {similarity_score:.3f}", color, "bold")
    print_colored(f"🏆 Метрика образования: {total_score:.2f}/1.00", color)
    print_colored(f"💡 РЕКОМЕНДАЦИЯ: {result['report']['analysis']['recommendation']}",
                  "green" if similarity_score >= 0.6 else "yellow")

    # Детальный анализ
    analysis = result['report']['analysis']
    print_separator("📊 ДЕТАЛЬНЫЙ АНАЛИЗ", 80, "-")

    # Технические навыки
    tech = analysis['technical_skills']
    tech_color = "green" if tech['score'] >= 0.7 else "yellow" if tech['score'] >= 0.5 else "red"
    print_colored(f"🔧 ТЕХНИЧЕСКИЕ НАВЫКИ: {tech['score']:.2f}", tech_color)
    print_colored(f"   ✅ Совпадений: {tech['matched_count']}/{tech['total_required']}",
                  "green" if tech['matched_count'] > 0 else "yellow")

    if tech['matched_skills']:
        print_colored(f"   🎯 Совпавшие: {', '.join(tech['matched_skills'][:5])}", "green")
    if tech['missing_skills']:
        print_colored(f"   ❌ Отсутствуют: {', '.join(tech['missing_skills'][:3])}", "red")
    if tech['extra_skills']:
        print_colored(f"   ➕ Дополнительные: {', '.join(tech['extra_skills'][:3])}", "cyan")

    # Опыт работы
    exp = analysis['experience']
    exp_color = "green" if exp['score'] >= 0.8 else "yellow" if exp['score'] >= 0.6 else "red"
    print_colored(f"\n💼 ОПЫТ РАБОТЫ: {exp['score']:.2f}", exp_color)
    print_colored(f"   📅 Кандидат: {exp['total_years']} лет", "white")
    print_colored(f"   🎯 Требуется: {exp['required_years']} лет", "white")

    if exp['total_years'] >= exp['required_years']:
        print_colored("   ✅ Достаточный опыт", "green")
    else:
        deficit = exp['required_years'] - exp['total_years']
        print_colored(f"   ⚠️  Не хватает {deficit} лет опыта", "yellow")

    # Образование
    # Образование
    edu = analysis['education']
    edu_color = "green" if edu['score'] >= 0.8 else "yellow" if edu['score'] >= 0.6 else "red"
    print_colored(f"\n🎓 ОБРАЗОВАНИЕ: {edu['score']:.2f}", edu_color)
    print_colored(f"   📚 Кандидат: {edu['highest_level'] or 'Не указано'}", "white")
    print_colored(f"   🎯 Требуется: {edu['required_level'] or 'Не указано'}", "white")

    # Добавьте детальную информацию об уровнях
    if edu.get('resume_level_value', 0) > 0 and edu.get('required_level_value', 0) > 0:
        if edu['resume_level_value'] >= edu['required_level_value']:
            print_colored("   ✅ Уровень образования соответствует требованиям", "green")
        else:
            deficit = edu['required_level_value'] - edu['resume_level_value']
            level_names = {1: 'школа', 2: 'среднее', 3: 'колледж', 4: 'бакалавр',
                           5: 'специалист', 6: 'магистр', 7: 'кандидат', 8: 'доктор'}
            required_name = level_names.get(edu['required_level_value'], 'требуемый уровень')
            current_name = level_names.get(edu['resume_level_value'], 'текущий уровень')
            print_colored(f"   ⚠️  Не хватает {deficit} уровня(ей) образования", "yellow")
            print_colored(f"   📉 Текущий: {current_name}, Требуется: {required_name}", "yellow")

    # Языковые навыки - ДЕТАЛЬНЫЙ АНАЛИЗ
    lang = analysis['language_skills']
    lang_color = "green" if lang['score'] >= 0.8 else "yellow" if lang['score'] >= 0.6 else "red"
    print_colored(f"\n🌍 ЯЗЫКОВЫЕ НАВЫКИ: {lang['score']:.2f}", lang_color)

    if lang['required_languages']:
        matched = len(lang['matched_languages'])
        required = len(lang['required_languages'])
        print_colored(f"   ✅ Совпадений: {matched}/{required} языков",
                      "green" if matched == required else "yellow")

        # Детали по требуемым языкам
        print_colored("   🎯 Требуемые языки:", "white")
        for lang_name, level in lang['required_languages'].items():
            status = "✅" if lang_name in [l.split(' (')[0] for l in lang.get('matched_languages', [])] else "❌"
            print_colored(f"      {status} {lang_name}: {level}",
                          "green" if status == "✅" else "red")

        if lang['matched_languages']:
            print_colored(f"   🗣️  Совпавшие: {', '.join(lang['matched_languages'][:5])}", "green")
    else:
        print_colored("   📝 Требований к языкам нет", "white")

    # Дополнительная информация о языках кандидата
    if lang.get('resume_languages'):
        print_colored("   📚 Языки в резюме:", "cyan")
        for lang_name, level in lang['resume_languages'].items():
            print_colored(f"      • {lang_name}: {level}", "cyan")

    # Предложения по улучшению
    if analysis['improvement_suggestions']:
        print_separator("💡 ПРЕДЛОЖЕНИЯ ПО УЛУЧШЕНИЮ", 80, "-")
        for i, suggestion in enumerate(analysis['improvement_suggestions'], 1):
            print_colored(f"   {i}. {suggestion}", "yellow")

    # Детальный отчет от LLM
    print_separator("📝 ДЕТАЛЬНЫЙ ОТЧЕТ HR-СПЕЦИАЛИСТА", 80, "-")
    print_colored(result['report']['detailed_report'], "white")

    print_separator("", 100, "=")


def print_system_metrics(results: List[Dict]):
    """Вывод системных метрик"""
    if not results:
        return

    total_scores = [result['report']['analysis']['total_score'] for result in results]
    tech_scores = [result['report']['analysis']['technical_skills']['score'] for result in results]
    exp_scores = [result['report']['analysis']['experience']['score'] for result in results]
    edu_scores = [result['report']['analysis']['education']['score'] for result in results]
    lang_scores = [result['report']['analysis']['language_skills']['score'] for result in results]

    print_separator("📈 СИСТЕМНЫЕ МЕТРИКИ И СТАТИСТИКА", 100)

    print_colored("📊 СРЕДНИЕ ПОКАЗАТЕЛИ:", "blue", "bold")
    print_colored(f"   🏆 Общий балл: {np.mean(total_scores):.3f} ± {np.std(total_scores):.3f}", "cyan")
    print_colored(f"   🔧 Технические навыки: {np.mean(tech_scores):.3f} ± {np.std(tech_scores):.3f}", "cyan")
    print_colored(f"   💼 Опыт работы: {np.mean(exp_scores):.3f} ± {np.std(exp_scores):.3f}", "cyan")
    print_colored(f"   🎓 Образование: {np.mean(edu_scores):.3f} ± {np.std(edu_scores):.3f}", "cyan")
    print_colored(f"   🌍 Языки: {np.mean(lang_scores):.3f} ± {np.std(lang_scores):.3f}", "cyan")

    # Распределение рекомендаций
    recommendations = [result['report']['analysis']['recommendation'] for result in results]
    rec_counts = {}
    for rec in recommendations:
        rec_counts[rec] = rec_counts.get(rec, 0) + 1

    print_colored(f"\n🎯 РАСПРЕДЕЛЕНИЕ РЕКОМЕНДАЦИЙ:", "blue", "bold")
    for rec, count in rec_counts.items():
        percentage = (count / len(recommendations)) * 100
        color = "green" if "Рекомендуем" in rec else "yellow" if "Рассмотреть" in rec else "red"
        print_colored(f"   {rec}: {count} кандидатов ({percentage:.1f}%)", color)

    # Анализ технических навыков
    avg_matched = np.mean([result['report']['analysis']['technical_skills']['matched_count'] for result in results])
    avg_required = np.mean([result['report']['analysis']['technical_skills']['total_required'] for result in results])

    print_colored(f"\n🔧 АНАЛИЗ ТЕХНИЧЕСКИХ НАВЫКОВ:", "blue", "bold")
    coverage = (avg_matched / avg_required * 100) if avg_required > 0 else 0
    coverage_color = "green" if coverage >= 70 else "yellow" if coverage >= 50 else "red"
    print_colored(f"   📊 Среднее совпадение навыков: {avg_matched:.1f}/{avg_required:.1f}", "white")
    print_colored(f"   🎯 Покрытие требований: {coverage:.1f}%", coverage_color)

    # Анализ опыта
    avg_candidate_exp = np.mean([result['report']['analysis']['experience']['total_years'] for result in results])
    avg_required_exp = np.mean([result['report']['analysis']['experience']['required_years'] for result in results])

    print_colored(f"\n💼 АНАЛИЗ ОПЫТА РАБОТЫ:", "blue", "bold")
    exp_ratio = (avg_candidate_exp / avg_required_exp) if avg_required_exp > 0 else 0
    exp_color = "green" if exp_ratio >= 1.0 else "yellow" if exp_ratio >= 0.8 else "red"
    print_colored(f"   📅 Средний опыт кандидатов: {avg_candidate_exp:.1f} лет", "white")
    print_colored(f"   🎯 Средний требуемый опыт: {avg_required_exp:.1f} лет", "white")
    print_colored(f"   📈 Соотношение опыта: {exp_ratio:.2f}", exp_color)

    # Анализ образования
    edu_levels = []
    for result in results:
        edu = result['report']['analysis']['education']
        if edu.get('resume_level_value', 0) > 0 and edu.get('required_level_value', 0) > 0:
            edu_levels.append((edu['resume_level_value'], edu['required_level_value']))

    if edu_levels:
        avg_resume_edu, avg_required_edu = zip(*edu_levels)
        edu_ratio = (sum(avg_resume_edu) / sum(avg_required_edu)) if sum(avg_required_edu) > 0 else 0
        edu_color = "green" if edu_ratio >= 1.0 else "yellow" if edu_ratio >= 0.8 else "red"

        print_colored(f"\n🎓 АНАЛИЗ ОБРАЗОВАНИЯ:", "blue", "bold")
        print_colored(f"   📊 Средний уровень кандидатов: {np.mean(avg_resume_edu):.1f}", "white")
        print_colored(f"   🎯 Средний требуемый уровень: {np.mean(avg_required_edu):.1f}", "white")
        print_colored(f"   📈 Соотношение уровней: {edu_ratio:.2f}", edu_color)

    # Детальный анализ языковых навыков
    lang_stats = []
    for result in results:
        lang_analysis = result['report']['analysis']['language_skills']
        if lang_analysis.get('required_languages'):
            matched = len(lang_analysis.get('matched_languages', []))
            required = len(lang_analysis['required_languages'])
            lang_stats.append((matched, required))

    if lang_stats:
        avg_matched_lang, avg_required_lang = zip(*lang_stats)
        lang_coverage = (sum(avg_matched_lang) / sum(avg_required_lang) * 100) if sum(avg_required_lang) > 0 else 0
        lang_color = "green" if lang_coverage >= 80 else "yellow" if lang_coverage >= 60 else "red"

        print_colored(f"\n🌍 АНАЛИЗ ЯЗЫКОВЫХ НАВЫКОВ:", "blue", "bold")
        print_colored(
            f"   📊 Среднее совпадение языков: {np.mean(avg_matched_lang):.1f}/{np.mean(avg_required_lang):.1f}",
            "white")
        print_colored(f"   🎯 Покрытие языковых требований: {lang_coverage:.1f}%", lang_color)

        # Самые частые требуемые языки
        all_required_langs = []
        for result in results:
            langs = result['report']['analysis']['language_skills'].get('required_languages', {})
            all_required_langs.extend(langs.keys())

        if all_required_langs:
            lang_counter = Counter(all_required_langs)
            print_colored("   🏆 Самые частые языковые требования:", "white")
            for lang_name, count in lang_counter.most_common(5):
                print_colored(f"      • {lang_name}: {count} вакансий", "cyan")

    # Анализ распределения баллов
    print_colored(f"\n📈 РАСПРЕДЕЛЕНИЕ БАЛЛОВ:", "blue", "bold")
    print_colored(
        f"   🔧 Технические навыки: {np.mean(tech_scores):.3f} (min: {np.min(tech_scores):.3f}, max: {np.max(tech_scores):.3f})",
        "cyan")
    print_colored(
        f"   💼 Опыт работы: {np.mean(exp_scores):.3f} (min: {np.min(exp_scores):.3f}, max: {np.max(exp_scores):.3f})",
        "cyan")
    print_colored(
        f"   🎓 Образование: {np.mean(edu_scores):.3f} (min: {np.min(edu_scores):.3f}, max: {np.max(edu_scores):.3f})",
        "cyan")
    print_colored(
        f"   🌍 Языки: {np.mean(lang_scores):.3f} (min: {np.min(lang_scores):.3f}, max: {np.max(lang_scores):.3f})",
        "cyan")

    # Процент кандидатов, удовлетворяющих порогу
    threshold = 0.6
    above_threshold = sum(1 for score in total_scores if score >= threshold)
    threshold_percentage = (above_threshold / len(total_scores)) * 100
    threshold_color = "green" if threshold_percentage >= 50 else "yellow" if threshold_percentage >= 30 else "red"

    print_colored(f"\n🎯 КАНДИДАТЫ ВЫШЕ ПОРОГА ({threshold}):", "blue", "bold")
    print_colored(f"   ✅ {above_threshold} из {len(total_scores)} кандидатов ({threshold_percentage:.1f}%)",
                  threshold_color)

    # Классификация качества кандидатов
    strong_candidates = sum(1 for result in results if result['report']['analysis']['total_score'] >= 0.8)
    good_candidates = sum(1 for result in results if 0.6 <= result['report']['analysis']['total_score'] < 0.8)
    moderate_candidates = sum(1 for result in results if 0.4 <= result['report']['analysis']['total_score'] < 0.6)
    weak_candidates = sum(1 for result in results if result['report']['analysis']['total_score'] < 0.4)

    print_colored(f"\n🏆 РАСПРЕДЕЛЕНИЕ КАЧЕСТВА КАНДИДАТОВ:", "blue", "bold")
    total = len(results)
    print_colored(f"   🥇 Сильные (≥0.8): {strong_candidates} кандидатов ({(strong_candidates / total * 100):.1f}%)",
                  "green" if strong_candidates > 0 else "white")
    print_colored(f"   🥈 Хорошие (0.6-0.8): {good_candidates} кандидатов ({(good_candidates / total * 100):.1f}%)",
                  "cyan" if good_candidates > 0 else "white")
    print_colored(
        f"   🥉 Умеренные (0.4-0.6): {moderate_candidates} кандидатов ({(moderate_candidates / total * 100):.1f}%)",
        "yellow" if moderate_candidates > 0 else "white")
    print_colored(f"   ⚠️  Слабые (<0.4): {weak_candidates} кандидатов ({(weak_candidates / total * 100):.1f}%)",
                  "red" if weak_candidates > 0 else "white")


def print_no_documents_message():
    """Сообщение об отсутствии документов"""
    print_separator("❌ ОШИБКА: ДОКУМЕНТЫ НЕ НАЙДЕНЫ", 100)
    print_colored("Для работы системы необходимо разместить файлы в следующих папках:", "red", "bold")
    print_colored("\n📄 РЕЗЮМЕ КАНДИДАТОВ:", "yellow")
    print_colored("   documents/cv/pdf/", "cyan")
    print_colored("   documents/cv/docx/", "cyan")
    print_colored("   documents/cv/rtf/", "cyan")

    print_colored("\n📋 ОПИСАНИЯ ВАКАНСИЙ:", "yellow")
    print_colored("   documents/vacancy/pdf/", "cyan")
    print_colored("   documents/vacancy/docx/", "cyan")
    print_colored("   documents/vacancy/rtf/", "cyan")

    print_colored("\n💡 Поддерживаемые форматы: PDF, DOCX, RTF", "white")


def print_processing_time(start_time: datetime):
    """Вывод времени обработки"""
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    print_separator("⏰ ВРЕМЯ ОБРАБОТКИ", 100)
    print_colored(f"🕒 Начало: {start_time.strftime('%H:%M:%S')}", "white")
    print_colored(f"🕒 Завершение: {end_time.strftime('%H:%M:%S')}", "white")
    print_colored(f"⏱️  Общее время: {processing_time:.2f} секунд",
                  "green" if processing_time < 60 else "yellow")


def main():
    """Основная функция для оценки кандидатов"""
    start_time = datetime.now()
    logger.info("Starting HR Candidate Evaluation System...")

    try:
        # Создаем структуру папок
        create_folder_structure()

        # Инициализация системы
        rag_system = RAGSystem()
        logger.info("System initialized successfully.")

        # Получаем информацию о документах
        docs_info = rag_system.get_documents_info()

        # Вывод системной информации
        print_system_info(docs_info)

        # Вывод списка документов
        print_documents_list(rag_system)

        # Вывод анализа доменов
        print_domain_analysis(rag_system)

        if docs_info['resumes_count'] == 0 or docs_info['vacancies_count'] == 0:
            print_no_documents_message()
            return

        # Вывод информации о процессе сопоставления
        print_matching_process_info(rag_system)

        # Оценка кандидатов
        print_colored("\n🔄 Начинаем анализ кандидатов...", "yellow", "bold")
        results = rag_system.evaluate_candidates(top_n=5)

        if not results:
            print_separator("❌ КАНДИДАТЫ НЕ НАЙДЕНЫ", 100)
            print_colored("Не найдено подходящих кандидатов для имеющихся вакансий.", "red")
            return

        # Вывод отчетов по кандидатам
        print_separator("🎯 РЕЗУЛЬТАТЫ ОЦЕНКИ КАНДИДАТОВ", 100)
        print_colored(f"📊 Найдено кандидатов: {len(results)}", "green", "bold")

        for i, result in enumerate(results, 1):
            print_candidate_report(result, i)

        # Вывод системных метрик
        print_system_metrics(results)

        # Вывод времени обработки
        print_processing_time(start_time)

        print_separator("✅ АНАЛИЗ ЗАВЕРШЕН", 100)
        print_colored("Система успешно завершила оценку кандидатов!", "green", "bold")

    except KeyboardInterrupt:
        print_colored("\n⏹️  Работа системы прервана пользователем", "yellow", "bold")
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        print_colored(f"❌ Критическая ошибка: {e}", "red", "bold")
    finally:
        print_colored(f"\n📝 Логи сохранены в файл: hr_system_{start_time.strftime('%Y%m%d_%H%M%S')}.log", "white")


if __name__ == "__main__":
    main()