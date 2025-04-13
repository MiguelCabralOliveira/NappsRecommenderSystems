# data_loading/Dockerfile

# Use the official AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Set the working directory in the container
WORKDIR /var/task

# Copy the requirements file first to leverage Docker cache
# Assumes Docker build context is the NappsRecommender directory (parent of data_loading)
COPY data_loading/requirements.txt ./requirements.txt

# Install Python dependencies
# Upgrade pip and use --no-cache-dir to reduce image size
# Add build essentials if any dependency needs compilation (like some versions of psycopg2, though binary should avoid this)
# RUN yum update -y && yum install -y gcc python3-devel postgresql-devel && yum clean all # Example if needed
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- KEY CHANGE REMAINS THE SAME ---
# Copy the *entire* data_loading directory into /var/task/data_loading
# This preserves the package structure relative to the parent directory.
COPY data_loading /var/task/data_loading/
# --- END KEY CHANGE ---

# Set the CMD to your *unified* handler function, *including the package path*
# Format: <package_name>.<module_name>.<function_name>
# The package is 'data_loading', the module is 'lambda_handlers', the function is 'unified_data_loader_handler'.
# --- UPDATED CMD ---
CMD [ "data_loading.lambda_handlers.unified_data_loader_handler" ]
# --- END UPDATED CMD ---