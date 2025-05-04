FROM debian:bullseye-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.9 \
        python3.9-venv \
        python3.9-distutils \
        python3-pip \
        build-essential \
        gcc \
        python3.9-dev \
        espeak-ng \
        curl \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set python3.9 as default python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

CMD ["/bin/bash"]
