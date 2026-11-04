FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data/credit_g.csv ./data/credit_g.csv
RUN pip install --no-cache-dir . && mkdir -p reports/figures && \
    PYTHONPATH=src python -m credit_risk.train && \
    useradd --system --no-create-home api && chown -R api /app/artifacts

USER api
ENV PYTHONPATH=src
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "credit_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
