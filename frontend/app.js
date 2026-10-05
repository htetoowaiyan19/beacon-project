const $ = (id) => document.getElementById(id);
let history = [], controller = null;
function bubble(role, text) {
  const element = document.createElement('article'); element.className = role; element.textContent = text;
  $('messages').appendChild(element); element.scrollIntoView({ block: 'end' }); return element;
}
async function health() {
  try {
    const response = await fetch('/api/health'); if (!response.ok) throw new Error('Health check failed');
    const { gpu } = await response.json();
    $('health').textContent = `${gpu.device} · ${gpu.is_loaded ? (gpu.has_lora ? 'Trained adapter loaded' : 'Base model loaded') : 'Model loads on first message'}\nVRAM: ${gpu.vram_allocated_gb} GB`;
  } catch (error) { $('health').textContent = error.message; }
}
$('clear').onclick = () => { if (controller) return; history = []; $('messages').replaceChildren(); $('metrics').textContent = ''; $('error').textContent = ''; };
$('stop').onclick = () => controller?.abort();
$('form').onsubmit = async (event) => {
  event.preventDefault(); if (controller) return;
  const message = $('prompt').value.trim(); if (!message) return;
  controller = new AbortController(); $('send').disabled = true; $('clear').disabled = true; $('stop').hidden = false;
  $('error').textContent = ''; $('metrics').textContent = ''; $('messages').querySelector('.welcome')?.remove();
  bubble('user', message); const reply = bubble('assistant', ''); $('prompt').value = '';
  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
      body: JSON.stringify({ message, history, use_base_model: $('base').checked, think: $('think').checked,
        temperature: Number($('temperature').value), max_new_tokens: Number($('tokens').value), system_prompt: $('system').value.trim() || null })
    });
    if (!response.ok) throw new Error(`Chat failed (${response.status}): ${await response.text()}`);
    await readUIMessageStream(response.body, (part) => {
      if (part.type === 'text-delta') { reply.textContent += part.delta; reply.scrollIntoView({ block: 'end' }); }
      if (part.type === 'data-metrics') { const m = part.data; $('metrics').textContent = `${m.total_tokens} tokens · ${m.tokens_per_second} tokens/s · ${m.elapsed_seconds}s`; }
    });
    history.push({ role: 'user', content: message }, { role: 'assistant', content: reply.textContent });
  } catch (error) { $('error').textContent = error.name === 'AbortError' ? 'Generation stopped. Partial reply was not added to conversation history.' : error.message;
  } finally { controller = null; $('send').disabled = false; $('clear').disabled = false; $('stop').hidden = true; health(); $('prompt').focus(); }
};
health();
