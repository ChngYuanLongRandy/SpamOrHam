FROM python:slim

# User Args
ARG REPO_DIR="."
ARG PROJECT_USER="Randy"
ARG HOME_DIR="/home/$PROJECT_USER"

WORKDIR $HOME_DIR

# Copy everything over
COPY . .

# Install conda environment
RUN pip install --upgrade pip && pip install -r api-requirements.txt && pip install -r tests-requirements.txt 

# Non-root 
RUN groupadd -g 2222 $PROJECT_USER && useradd -u 2222 -g 2222 -m $PROJECT_USER
RUN chown -R 2222:2222 $HOME_DIR && \
    rm /bin/sh && ln -s /bin/bash /bin/sh
USER 2222

# Expose port for API
EXPOSE 8000

# CMD to execute to run Fastapi endpoint
CMD [ "uvicorn", "api.api.main:APP" , "--host", "0.0.0.0", "--port", "8000"]