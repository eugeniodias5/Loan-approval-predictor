FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install .

COPY app/ /app/

EXPOSE 8000

CMD ["uvicorn", "app.loan_pred_app:app", "--host", "0.0.0.0", "--port", "8000"]