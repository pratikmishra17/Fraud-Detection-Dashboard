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

# 5. (Optional) expose the default Streamlit port
EXPOSE 8501

# 6. Launch Streamlit, binding to the port Vercel provides in $PORT
CMD ["bash", "-lc", "streamlit run app.py --server.port $PORT --server.address=0.0.0.0"]
