# Gunakan base image Python
FROM python:3.8-slim

# Set working directory di dalam container
WORKDIR /app

# Salin file dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . .

# Download NLTK data di container
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab')"

EXPOSE 8080

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
