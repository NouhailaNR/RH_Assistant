from rag import Rag
from evaluatellm import Evaluatellm
import os
from langchain_chroma import Chroma


def main():

    persist_directory="chromaDB"
    collection_name="ma_collection"
    pdf_folder = "CVthèque"

    rag = Rag(collection_name=collection_name,persist_directory=persist_directory)
    eval=Evaluatellm()

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    try:
        base = Chroma(
            collection_name=collection_name,
            embedding_function=rag.embedding_model,
            persist_directory=persist_directory,
            
        )
        # Vérifier si la collection contient des documents
        docs_count = base._collection.count()
        if docs_count > 0:
            print(f"La collection '{collection_name}' existe déjà avec {docs_count} documents.")
        else:
            print(f"La collection '{collection_name}' est vide. Ajout des documents...")
            raise FileNotFoundError  # On passe à l'ajout des documents
    except Exception:
        print("Création d'une nouvelle base...")
        all_chunks = rag.chunking(pdf_folder)
        base = rag.create_vector_store(
            data=all_chunks,
            persist_directory=persist_directory,
            model=rag.embedding_model,
            collection_name=collection_name,)
    
    
    #verifier le résultat sur le vector_store
    query=input("Comment puis-je vous aidez aujourd'hui ? :")
    info_pertin = base.similarity_search(query, k=3)
    results=rag.ask_llm(rag.LLM_model,query,info_pertin)
    expected=input("quelle est la réponse attendue ?")
    # Évaluer la réponse
    eval_result = eval.evaluate_LLM(query, expected, results)
    
# Afficher l'évaluation
    print("Score :", eval_result["score"])
    print("Explication :", eval_result["reason"])
    print("\n================================= RÉPONSE DU LLM ===================================")
    print(results)
    print("===================================================================\n")
if __name__=="__main__":
    main()