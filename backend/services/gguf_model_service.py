"""Adapt local llama.cpp streaming to the existing chat and operator API."""
from __future__ import annotations
import json
import threading
import time
from backend.config import DEFAULT_SYSTEM_PROMPT
from backend.services.chat_context import ChatContextError
from scripts.utils.llama_runtime import LlamaRuntime


class GGUFModelService:
    def __init__(self, runtime=None):
        self.runtime = runtime or LlamaRuntime()
        self.lock = threading.Lock()
        self.max_output = 192
        self.max_prompt = 1536

    def load(self):
        self.runtime.start()

    def close(self):
        self.runtime.close()

    def get_gpu_status(self):
        loaded = self.runtime.ready and self.runtime.process is not None and self.runtime.process.poll() is None
        buffers = dict(self.runtime.buffers)
        # These are reported buffers, not total RAM usage or GPU utilization.
        memory = round(sum(buffers.values()) / 1024, 2) if buffers else None
        resident = available = None
        if loaded:
            try:
                import psutil
                resident = round(psutil.Process(self.runtime.process.pid).memory_info().rss / 1024**3, 2)
                available = round(psutil.virtual_memory().available / 1024**3, 2)
            except Exception:
                # Optional telemetry must survive a process exiting during a read.
                pass
        return {'device': 'Apple GPU (Metal)' if getattr(self.runtime, 'metal', False) else 'llama.cpp CPU',
                'runtime': 'llama.cpp', 'is_loaded': loaded, 'has_lora': loaded,
                'adapter_merged': True, 'adapter_available': True,
                'adapter_name': 'beacon-v1.0.0 merged / ' + getattr(self.runtime, 'metadata', {}).get('quantization', 'Q4_K_M'), 'cpu_offload': False,
                'execution_devices': ['Metal' if getattr(self.runtime, 'metal', False) else 'CPU'],
                'shared_memory': True, 'runtime_buffers_gb': memory,
                'native_process_ram_gb': resident, 'system_available_ram_gb': available,
                'vram_allocated_gb': None, 'vram_reserved_gb': None,
                'max_new_tokens': self.max_output, 'max_prompt_tokens': self.max_prompt}

    def prepare(self, messages, system_prompt):
        if not messages or messages[-1].get('role') != 'user':
            raise ChatContextError('A conversation must end with your current message.')
        groups = []
        for message in messages[:-1]:
            if message['role'] == 'user':
                groups.append([message])
            elif groups:
                groups[-1].append(message)
        selected = groups[-2:]
        while True:
            conversation = [{'role': 'system', 'content': system_prompt}]
            conversation += [m for group in selected for m in group] + [messages[-1]]
            prompt = self.runtime.json('/apply-template', {
                'messages': conversation, 'chat_template_kwargs': {'enable_thinking': False}})['prompt']
            tokens = self.runtime.json('/tokenize', {
                'content': prompt, 'add_special': True, 'parse_special': True})['tokens']
            if len(tokens) <= self.max_prompt:
                return prompt, len(tokens), dict(history_turns_used=len(selected),
                    history_messages_dropped=len(messages) - 1 - sum(map(len, selected)),
                    prompt_token_budget=self.max_prompt)
            if not selected:
                raise ChatContextError('Your message and instructions are too long for laptop chat. Shorten them and try again.')
            selected.pop(0)

    def stream_chat(self, *, messages, use_base_model=False, think=False, temperature=.7,
                    top_p=.8, max_new_tokens=192, system_prompt=None):
        if use_base_model:
            raise ChatContextError('The laptop model contains the trained adapter merged into its weights. Base comparison is unavailable.')
        if think:
            raise ChatContextError('Thinking mode is disabled in the laptop edition to keep replies responsive.')
        queued = time.perf_counter()
        with self.lock:
            queue_seconds = time.perf_counter() - queued
            self.load()
            system = system_prompt or DEFAULT_SYSTEM_PROMPT
            current = messages[-1]['content']
            if not system_prompt and current.isascii() and any(c.isalpha() for c in current):
                system += ' The current user message is in English. Reply in English unless another language is requested.'
            prompt, prompt_tokens, metrics = self.prepare(messages, system)
            limit = min(max_new_tokens, self.max_output)
            started = time.perf_counter()
            final = None
            payload = {'prompt': prompt, 'n_predict': limit, 'temperature': temperature,
                       'top_p': top_p, 'repeat_penalty': 1.15, 'cache_prompt': True, 'stream': True}
            with self.runtime.request('/completion', payload) as response:
                for raw in response:
                    line = raw.decode('utf-8').strip()
                    if not line.startswith('data: '):
                        continue
                    if line[6:] == '[DONE]':
                        break
                    frame = json.loads(line[6:])
                    if 'error' in frame:
                        raise RuntimeError('llama.cpp generation failed')
                    if frame.get('content'):
                        yield frame['content']
                    if frame.get('stop'):
                        final = frame
            if final is None:
                raise RuntimeError('The model stream ended before completion')
            elapsed = time.perf_counter() - started
            count = final.get('tokens_predicted', final.get('timings', {}).get('predicted_n', 0))
            return {**metrics, 'prompt_tokens': prompt_tokens, 'total_tokens': count,
                    'elapsed_seconds': round(elapsed, 3), 'tokens_per_second': round(count / elapsed, 1) if elapsed else 0,
                    'queue_seconds': round(queue_seconds, 3), 'effective_max_new_tokens': limit,
                    'finish_reason': 'length' if final.get('stop_type') == 'limit' or count >= limit else 'stop'}
