from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
#from langchain.evaluation.schema import EvalResult
import os
import re
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.genai.errors import ClientError


load_dotenv()

class Rag():

    def __init__(self, api_key=None, collection_name="cv_collection", persist_directory="db/chroma"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.documents=[]
        self.collection_name=collection_name
        self.presist_directory=persist_directory  
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        self.LLM_model=GoogleGenerativeAI(model="models/gemini-2.5-pro")
    
    # 1) extraction et nettoyer mon texte

    def extract_text_with_pypdf2(self,pdf_folder):
        pdfs = {}
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                path = os.path.join(pdf_folder, filename)
                text=""
                reader=PdfReader(path)
                for page in reader.pages[1:]:
                    page_text = page.extract_text()
                    if page_text:  # vérifie qu'il y a du texte
                        text += page_text + "\n"
                pdfs[filename]=text
        return pdfs

    def clean(self, text):
        text = re.sub(r"[#*/\[\]]", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text).strip()
        return text

    # 2) chunking
    def chunking(self,pdf_folder):  
        all_chunks=[]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        pdfs = self.extract_text_with_pypdf2(pdf_folder)
        for fname, pdf in pdfs.items():
            text=self.clean(pdf)
            candidat_name=fname.replace("CV_","").replace(".pdf","")
            chunks = splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    Document(
                        page_content=chunk,metadata={"candidate":candidat_name,  "chunk_id":i, **pdf.metadata}))
        return all_chunks

    # 3) embedding et stockage de vector store dans base des données

    def create_vector_store(self,data,persist_directory,model,collection_name):
        """Charger la base Chroma """
        self.documents=data
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=model,
            persist_directory=persist_directory
            )
        vector_store.add_documents(self.documents)
        return vector_store   

    #4) recuperer les vectorstor et appliquer le Rag
    @retry(
        retry=retry_if_exception_type(ClientError),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(5)
    )
    def ask_llm(self,llm,question,docs):
        """
        Pose une question à ta base Chroma en utilisant LLM et RAG.
        
        Args:
            question (str): la question de l'utilisateur
            chroma_path (str): chemin vers la base Chroma persistante
            k (int): nombre de chunks à récupérer

        Redevturns:
            str: réponse générée par le LLM
        """

        # création de prompt
        context = "\n\n".join([d.page_content for d in docs])
        prompt_template ="""

            Tu es un assistant expert en analyse de CV et en réponse à des questions spécifiques.
            Ton rôle:
            -utiliser uniquement les informations contenues dans les CV fournis.
            - répondre de manière structurée, claire et complète, en reformulant si nécessaire.
            - Ne jamais inventer ou supposer des informations.

            Contexte (CVs disponible):
            {context}

            Question :
            {question}

            Instructions :
            1. Analyse la question pour déterminer si l'utilisateur cherche :
            - Une seule personne → sélectionne uniquement le CV le plus pertinent.
            - Plusieurs personnes → sélectionne exactement le nombre demandé de CVs les plus pertinents.
            2. Tu peux reformuler, synthétiser et structurer les informations:
            - Ne mentionne JAMAIS les CVs non retenus.
            3. N’inclut PAS l’intégralité des CV dans la réponse.
            4. Si aucun CV n’est pertinent, indique-le clairement réponds : « Aucun CV pertinent trouvé. ».

            Format attendu :
            Réponse : 
            - Analyse claire et complète.
            - Reformulation autorisée.
            Justification :
            - Liste précise des éléments utilisés, tirés directement des CV retenus.
            """

        prompt=PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"])
        
        # Poser la question
        chain = prompt | llm
        response = chain.invoke({"context": context,"question": question})
        return response

