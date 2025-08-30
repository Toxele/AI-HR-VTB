from document_managament.pdf_document import Pdfdocument
from document_managament.docx_document import DocxDocument
from document_managament.rtf_document import RtfDocument
from typing import Union


class DocumentFactory:
    @staticmethod
    def create_document(
            document_name: str,
            document_dir: str,
            extract_dir_name: str,
            text_embeddings=None,
            rebuild_index: bool = False
    ) -> Union[Pdfdocument, DocxDocument, RtfDocument, None]:
        """Фабрика для создания документов разных типов"""
        file_extension = document_name.lower().split('.')[-1]

        if file_extension == 'pdf':
            return Pdfdocument(
                document_name=document_name,
                document_dir=document_dir,
                extract_dir_name=extract_dir_name,
                text_embeddings=text_embeddings,
                rebuild_index=rebuild_index
            )
        elif file_extension == 'docx':
            return DocxDocument(
                document_name=document_name,
                document_dir=document_dir,
                extract_dir_name=extract_dir_name,
                text_embeddings=text_embeddings,
                rebuild_index=rebuild_index
            )
        elif file_extension == 'rtf':
            return RtfDocument(
                document_name=document_name,
                document_dir=document_dir,
                extract_dir_name=extract_dir_name,
                text_embeddings=text_embeddings,
                rebuild_index=rebuild_index
            )
        else:
            print(f"Неподдерживаемый формат файла: {file_extension}")
            return None