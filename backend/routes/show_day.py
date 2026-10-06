"""Token-protected operator feed; visitor events contain no draft text."""
import hmac
from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from backend.release import public_release

router = APIRouter(prefix='/api', tags=['Show day'])


class VisitorEvent(BaseModel):
    event: Literal['connected', 'heartbeat', 'typing', 'input_focused', 'new_chat', 'stop', 'copy', 'save']
    client_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    session_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')


def operator_monitor(request):
    monitor = request.app.state.show_day_monitor
    if monitor is None:
        raise HTTPException(404, 'Show-day monitoring is disabled')
    supplied = request.headers.get('authorization', '').removeprefix('Bearer ')
    if not hmac.compare_digest(supplied.encode(), monitor.token.encode()):
        raise HTTPException(401, 'Operator token required')
    return monitor


@router.post('/visitor/event')
def visitor_event(event: VisitorEvent, request: Request):
    monitor = request.app.state.show_day_monitor
    if monitor is None:
        raise HTTPException(404, 'Show-day monitoring is disabled')
    monitor.activity(event.event, event.client_id, event.session_id)
    return {'ok': True}


@router.get('/operator/snapshot')
def operator_snapshot(request: Request, after: int = 0):
    monitor = operator_monitor(request)
    snapshot = monitor.snapshot(after)
    snapshot['gpu'] = request.app.state.chat_service.model_service.get_gpu_status()
    snapshot['release'] = public_release()
    return snapshot


@router.post('/operator/shutdown')
def operator_shutdown(request: Request):
    operator_monitor(request)
    stop = getattr(request.app.state, 'show_day_shutdown', None)
    if stop is None:
        raise HTTPException(409, 'This server was not started by the show-day launcher')
    stop()
    return {'status': 'stopping'}
