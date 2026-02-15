FROM python:3.11-slim

WORKDIR /app

# Install system dependency for LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Copy requirements first (for better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

# Copy application code
COPY flask_app/ .

# Copy model + vectorizer
COPY tfidf_vectorizer.pkl .
COPY lgbm_model.pkl .

EXPOSE 5000

CMD ["python", "app.py"]
