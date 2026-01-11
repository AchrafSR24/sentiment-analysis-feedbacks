FROM python:3.10-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger les ressources NLTK
RUN python -m nltk.downloader vader_lexicon stopwords wordnet omw-1.4

# Copier le projet
COPY . .

# Exposer le port Streamlit
EXPOSE 8501

# Lancer l'app Streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
