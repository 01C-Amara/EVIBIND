FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY evibind ./evibind
COPY tapbench ./tapbench
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 evibind
USER evibind

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=2)"

CMD ["evibind", "serve", "--host", "0.0.0.0", "--port", "8090"]
