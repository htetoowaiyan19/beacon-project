"""Direct streaming test script."""

import sys
import httpx
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    client = httpx.Client(timeout=180.0)
    payload = {
        "message": "Beacon AI ရဲ့ အဓိက စွမ်းဆောင်ရည်တွေကို အကျဉ်းချုပ် ရှင်းပြပေးပါ။",
        "history": [],
        "max_new_tokens": 80,
        "think": False,
    }

    print("Sending request to /api/chat/stream...")
    with client.stream("POST", "http://127.0.0.1:8000/api/chat/stream", json=payload) as response:
        print(f"Response status: {response.status_code}")
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                if line[6:] == "[DONE]":
                    break
                payload = json.loads(line[6:])
                evt_type = payload.get("type")
                if evt_type == "text-delta":
                    print(payload.get("delta", ""), end="", flush=True)
                elif evt_type == "data-metrics":
                    print("\n\n[Done Metrics]:", payload.get("data"))


if __name__ == "__main__":
    main()
