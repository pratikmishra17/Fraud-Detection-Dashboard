# Dockerfile

# 1. Use a minimal Python base image
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your entire app
COPY . .

# 5. Launch Streamlit on Vercel’s assigned port
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
