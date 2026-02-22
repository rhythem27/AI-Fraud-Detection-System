from celery import Celery
import os

# Initialize Celery
celery_app = Celery(
    "fraud_detection_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Optional configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_publish_retry=True,
)

# Auto-discover tasks from the services directory
celery_app.autodiscover_tasks(["services"])
