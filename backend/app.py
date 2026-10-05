"""Default entrypoint for model-only chat."""
from backend.application import create_app

app = create_app()
