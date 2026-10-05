"""Explicit frontend-only preview using real routes and synthetic streaming text."""
import time
from backend.application import create_app
from backend.services.chat_service import ChatService


class PreviewModel:
    def get_gpu_status(self):
        return dict(device='Frontend preview (mock; no GPU)', is_loaded=False,
                    has_lora=False, vram_allocated_gb=0, vram_reserved_gb=0)

    def stream_chat(self, messages, **options):
        text = ('[Frontend preview — synthetic reply]\n'
                'မင်္ဂလာပါ။ ဒါက စာသားစီးဆင်းမှု စမ်းသပ်ရန် နမူနာအဖြေ ဖြစ်ပါတယ်။\n'
                'Your message: ' + messages[-1]['content'])
        started = time.monotonic()
        for offset in range(0, len(text), 8):
            time.sleep(.03)
            yield text[offset:offset+8]
        return dict(total_tokens=0, prompt_tokens=0, elapsed_seconds=round(time.monotonic()-started,2),
                    tokens_per_second=0, finish_reason='stop', synthetic=True)


app = create_app(chat_service=ChatService(model_service=PreviewModel()))
