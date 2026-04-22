import os
from celery import Celery
from .ingestor import CORD19Ingestor
from .db import SessionLocal
from embeddings.engine import EmbeddingEngine

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery("covid_tasks", broker=REDIS_URL)

@app.task
def ingest_cord19_task(limit: int = 1000):
    db = SessionLocal()
    try:
        ingestor = CORD19Ingestor(db)
        ingestor.fetch_cord19(limit=limit)
    finally:
        db.close()

@app.task
def update_embeddings_task(batch_size: int = 100):
    db = SessionLocal()
    try:
        engine = EmbeddingEngine()
        engine.update_paper_embeddings(db, batch_size=batch_size)
    finally:
        db.close()

# Periodic task setup (optional)
app.conf.beat_schedule = {
    'update-embeddings-every-hour': {
        'task': 'data_pipeline.tasks.update_embeddings_task',
        'schedule': 3600.0,
        'args': (100,)
    },
}
