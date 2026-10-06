"""Close the model iterator deterministically when an HTTP client disconnects."""
import anyio
from fastapi.responses import StreamingResponse


class ClosingStreamingResponse(StreamingResponse):
    def __init__(self, source, **kwargs):
        self.source = source
        super().__init__(source, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette's threadpool iterator does not close its sync source.
            # Shield cleanup so Stop/disconnect reaches the model cancellation event.
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    await anyio.to_thread.run_sync(self.source.close)
