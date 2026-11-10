# Image qui contient tout le projet et lance l'application web.
# Construire : docker build -t qualite-vin .
# Lancer     : docker run -p 5000:5000 qualite-vin     puis ouvrir http://localhost:5000
FROM python:3.13-slim

WORKDIR /projet
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Le modèle est réentraîné pendant la construction : l'image ne dépend d'aucun fichier préparé à la main.
RUN python -m src.entrainement

EXPOSE 5000
CMD ["python", "app/app.py"]
