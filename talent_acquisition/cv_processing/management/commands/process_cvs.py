import os
from django.core.management.base import BaseCommand
from cv_processing.models import CV
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from pdfminer.high_level import extract_text

class Command(BaseCommand):
    help = 'Process CVs in a directory'

    def handle(self, *args, **kwargs):
        directory = input("Enter the directory containing CVs: ")
        for file in os.listdir(directory):
            if file.endswith(".pdf"):
                path = os.path.join(directory, file)
                text = extract_text(path)
                # Generate embeddings
                embedding_model = OpenAIEmbeddings()
                embeddings = embedding_model.embed(text)
                # Save to Milvus and Django DB
                Milvus().add_documents([embeddings])
                CV.objects.create(name=file, file=path, embeddings=embeddings)
                print(f"Processed {file}")
