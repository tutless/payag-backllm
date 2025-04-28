FROM python:3.12-slim

# Set up working directory
WORKDIR /server

# Install pipenv and remove cache after installation
RUN pip install --upgrade pip && pip install --no-cache-dir pipenv

# Copy dependency files first to leverage Docker layer caching
COPY Pipfile Pipfile.lock ./

# Prevent Pipenv from creating a virtual environment inside Docker
ENV PIPENV_VENV_IN_PROJECT=1

# Install only necessary dependencies
RUN pipenv install --deploy --ignore-pipfile --system && \
    rm -rf /root/.cache/pip /root/.local/share/virtualenvs

# Copy the rest of the application code
COPY . .

# Set up a volume (optional, only if data_source is needed)

# Expose port

EXPOSE 5888

EXPOSE 51150

# Run the application
CMD ["python", "server.py"]

