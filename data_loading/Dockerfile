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
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- KEY CHANGE ---
# Copy the *entire* data_loading directory into /var/task/data_loading
# This preserves the package structure relative to the parent directory.
COPY data_loading /var/task/data_loading/
# --- END KEY CHANGE ---

# Set the CMD to your handler function, *including the package path*
# Format: <package_name>.<module_name>.<function_name>
# The package is now 'data_loading' because we copied the whole directory.
#
# **IMPORTANT**: If you intend this Lambda function to run the Shopify trigger,
# change 'db_loader_handler' to 'shopify_trigger_handler' below.
CMD [ "data_loading.lambda_handlers.db_loader_handler" ]