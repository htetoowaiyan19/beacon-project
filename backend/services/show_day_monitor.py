"""Bounded, in-memory seminar activity; no chat content is written to disk."""
from collections import OrderedDict, deque
from copy import deepcopy
import json
from threading import RLock
import time
from uuid import uuid4


class ShowDayMonitor:
    def __init__(self, token, max_chats=40, max_events=250):
        self.token = token
        self.max_chats = max_chats
        self.started = time.time()
        self.lock = RLock()
        self.chats = OrderedDict()
        self.events = deque(maxlen=max_events)
        self.visitors = OrderedDict()
        self.sequence = 0
        self.total = self.completed = self.failed = self.cancelled = self.active = 0
        self.latest_metrics = {}

    def _event(self, kind, client_id, session_id, **fields):
        self.sequence += 1
        self.events.append(dict(seq=self.sequence, at=time.time(), type=kind,
                                client_id=client_id, session_id=session_id, **fields))
        self.visitors[client_id] = dict(client_id=client_id, session_id=session_id,
                                        action=kind, last_seen=time.time())
        self.visitors.move_to_end(client_id)
        while len(self.visitors) > 100:
            self.visitors.popitem(last=False)

    def activity(self, kind, client_id, session_id):
        with self.lock:
            self._event(kind, client_id, session_id)

    def track(self, generator, *, message, client_id=None, session_id=None, use_base_model=False):
        chat_id = uuid4().hex
        client_id = client_id or 'api-client'
        session_id = session_id or chat_id
        with self.lock:
            self.total += 1; self.active += 1
            chat = dict(id=chat_id, client_id=client_id, session_id=session_id,
                        message=message, answer='', status='waiting', started_at=time.time(),
                        model='base' if use_base_model else 'trained', metrics={})
            self.chats[chat_id] = chat
            while len(self.chats) > self.max_chats:
                self.chats.popitem(last=False)
            self._event('chat_started', client_id, session_id, chat_id=chat_id)
        finished = False
        finish_seen = False
        failure = False
        try:
            for frame in generator:
                for line in frame.splitlines():
                    if line == 'data: [DONE]':
                        with self.lock:
                            finished = finish_seen
                            chat['status'] = ('failed' if failure else 'complete') if finished else 'cancelled'
                        continue
                    if not line.startswith('data: '):
                        continue
                    part = json.loads(line[6:])
                    with self.lock:
                        if part['type'] == 'text-delta':
                            chat['answer'] = (chat['answer'] + part['delta'])[:64000]
                            chat['status'] = 'replying'
                        elif part['type'] == 'data-metrics':
                            chat['metrics'] = part['data']
                            self.latest_metrics = part['data']
                        elif part['type'] == 'error':
                            failure = True
                            chat['error'] = part.get('errorText', 'Generation failed')
                        elif part['type'] == 'finish':
                            failure = failure or part.get('finishReason') == 'error'
                            finish_seen = True
                            chat['status'] = 'finishing'
                            chat['finish_reason'] = part.get('finishReason', 'stop')
                yield frame
        except Exception:
            failure = finished = True
            chat['status'] = 'failed'
            raise
        finally:
            try:
                generator.close()
            finally:
                with self.lock:
                    self.active -= 1
                    if not finished:
                        chat['status'] = 'cancelled'
                        self.cancelled += 1
                    elif failure:
                        self.failed += 1
                    else:
                        self.completed += 1
                    chat['ended_at'] = time.time()
                    self._event('chat_' + chat['status'], client_id, session_id, chat_id=chat_id)

    def snapshot(self, after=0):
        with self.lock:
            now = time.time()
            return deepcopy(dict(
                sequence=self.sequence, uptime_seconds=round(now - self.started),
                counters=dict(total=self.total, completed=self.completed, failed=self.failed,
                              cancelled=self.cancelled, active=self.active),
                latest_metrics=self.latest_metrics,
                visitors=[dict(v, online=now - v['last_seen'] < 35) for v in self.visitors.values()],
                chats=list(self.chats.values()),
                events=[e for e in self.events if e['seq'] > after],
            ))
