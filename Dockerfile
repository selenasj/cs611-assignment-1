FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jdk-headless procps bash \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /bin/bash /bin/sh

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${PATH}:${JAVA_HOME}/bin"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]

