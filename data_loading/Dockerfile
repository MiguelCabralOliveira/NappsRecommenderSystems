FROM public.ecr.aws/lambda/python:3.10

# Set common Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Set the working directory within the container
WORKDIR ${LAMBDA_TASK_ROOT} # /var/task

# Copiar requirements.txt (assumindo que está em NappsRecommender/data_loading/)
# O contexto de build deve ser NappsRecommender/
COPY data_loading/requirements.txt .

# Instalar dependências
# Consider using --platform manylinux2014_x86_64 for compiled libraries if needed, though base lambda image should handle it
RUN pip install -r requirements.txt --no-cache-dir

# Copiar TODO o conteúdo de data_loading/ para a raiz do WORKDIR (/var/task)
# Isso inclui lambda_handlers.py, run_loading.py, utils.py, database/, shopify/
COPY data_loading/ .

# Definir o handler
# O arquivo é 'lambda_handlers.py' (agora em /var/task/lambda_handlers.py)
# A função dentro dele é 'db_loader_handler' (ou 'shopify_trigger_handler')
# Altere abaixo para o handler que você quer que este container execute por padrão
CMD [ "lambda_handlers.db_loader_handler" ]
# Para o outro handler, você pode sobrescrever o CMD ou criar outra imagem/lambda
# CMD [ "lambda_handlers.shopify_trigger_handler" ]