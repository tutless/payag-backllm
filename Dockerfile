# Use CUDA runtime image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Install system dependencies and Python
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    tzdata \
    curl \
    gnupg \
    ca-certificates \
    lsb-release \
    build-essential && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade pip pipenv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Set working directory
WORKDIR /llmsvr

# Keep virtualenv inside project dir
ENV PIPENV_VENV_IN_PROJECT=true

# Copy dependency files first to leverage Docker cache
COPY Pipfile Pipfile.lock ./

# Patch: relax the python_version in Pipfile (handled next)
# You need to update your local Pipfile like so:
# [requires]
# python_version = "3.10"
# Then regenerate Pipfile.lock with:
#   pipenv lock

# Ensure Python 3.10 is available for Pipenv
ENV PATH="/usr/bin:$PATH"

# Install dependencies inside Pipenv virtualenv
RUN pipenv install --deploy --ignore-pipfile --python /usr/bin/python3.10

RUN pipenv run pip install sentence-transformers

# Show installed packages (optional)
RUN pipenv run pip list

# Add venv to path for shell access
ENV PATH="/llmsvr/.venv/bin:$PATH"

# Copy the full application
COPY . .

# Set CUDA library path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib/x86_64-linux-gnu

# Expose port

EXPOSE 5888

EXPOSE 51150

# Default command
CMD ["pipenv", "run", "python", "server.py"]
