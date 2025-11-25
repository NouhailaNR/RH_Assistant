# LLM – RAG Assistant RH

Ce projet implémente une application RAG (Retrieval Augmented Generation) destinée à créer un assistant RH intelligent capable de :
- Analyser et comprendre le contenu de CVs au format PDF
- Répondre à des questions sur les candidats en utilisant un modèle LLM
- Identifier le profil le plus adapté à un poste donné
- Évaluer la qualité des réponses générées grâce à un module d’évaluation automatique (DeepEval)

- `rag.py`:	Contient la classe Rag, responsable de l'extraction de texte, nettoyage, chunking, génération d'embeddings, stockage vectoriel avec ChromaDB et génération de réponses via RAG.
-`evaluatellm.py`:	Contient la classe permettant d’évaluer la réponse du RAG par rapport à une réponse attendue (score).
- `main.py`:	Script principal permettant d’exécuter l’application en terminal.

---

## Utilisation :
Ce projet utilise un environnement virtuel Python pour gérer les dépendances.

Les dépendances sont listées dans les fichiers :

    env/requirements.txt: pour utilisation avec pip
    env/environment.yml: pour utilisation avec conda

Pour installer tous les packages listés dans ce fichier, utilisez la commande :

    pip install -r env/requirements.txt ou alors
    conda create -f env/environment.yml selon si vous utilisez pip ou conda.


#### Excution de code:

python main.py
