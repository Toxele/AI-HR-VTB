import os
from pathlib import Path


def create_folder_structure():
    """Создает структуру папок для документов"""
    base_folders = ['cv', 'vacancy']
    sub_folders = ['pdf', 'docx', 'rtf']

    for base in base_folders:
        for sub in sub_folders:
            folder_path = Path('documents') / base / sub
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Создана папка: {folder_path}")

    print("Структура папок создана успешно!")


if __name__ == "__main__":
    create_folder_structure()