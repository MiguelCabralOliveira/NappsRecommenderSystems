# preprocessing/Dockerfile

# 1. Base Image
FROM python:3.10-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Set Working Directory
WORKDIR /app

# 5. Copy the specific requirements file for preprocessing
COPY ./preprocessing/requirements.txt /app/requirements.txt

# 6. Install Python Dependencies for preprocessing ONLY
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the entire preprocessing module content into the image's workdir
COPY ./preprocessing /app/preprocessing

# 8. Define o Entrypoint para executar o script run_preprocessing.py como módulo'
ENTRYPOINT ["python", "-m", "preprocessing.run_preprocessing"]

# 9. Default CMD (args virão do SageMaker)
CMD []