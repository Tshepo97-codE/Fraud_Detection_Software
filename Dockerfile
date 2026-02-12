FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for caching
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# EXPLICITLY INSTALL UVICORN
RUN pip install uvicorn[standard]

# Copy application code
COPY api/ ./api/
COPY models/ ./models/

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"] 