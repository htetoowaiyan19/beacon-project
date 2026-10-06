const $ = id => document.getElementById(id);
let history = [], controller = null;
const welcome = $('messages').innerHTML;
function scrollChat() { $('messages').scrollTop = $('messages').scrollHeight; }
function bubble(role, text) {
 const card = document.createElement('article'); card.className = 'message ' + role;
 const meta = document.createElement('div'); meta.className = 'message-meta';
 const name = document.createElement('span'); name.textContent = role === 'user' ? 'YOU' : $('base').checked ? 'QWEN · BASE MODEL' : 'BEACON'; meta.appendChild(name);
 const body = document.createElement('div'); body.className = 'message-body'; body.textContent = text;
 const copy = document.createElement('button'); copy.className = 'copy'; copy.type = 'button'; copy.textContent = 'Copy'; copy.setAttribute('aria-label', 'Copy ' + role + ' message');
 copy.onclick = async () => { try { await navigator.clipboard.writeText(body.textContent); copy.textContent = 'Copied'; setTimeout(() => copy.textContent = 'Copy', 1500); } catch { $('error').textContent = 'Copy is unavailable. Select the message text to copy it.'; } };
 meta.appendChild(copy); card.append(meta, body); $('messages').appendChild(card); scrollChat(); return { card, body };
}
async function health() {
 try {
  const response = await fetch('/api/health'); if (!response.ok) throw new Error('Health check failed');
  const { gpu, release } = await response.json();
  if (gpu.runtime === 'llama.cpp') {
   $('base').checked = false; $('base').disabled = true;
   $('think').checked = false; $('think').disabled = true;
   $('tokens').max = gpu.max_new_tokens;
   if (Number($('tokens').value) > gpu.max_new_tokens) $('tokens').value = gpu.max_new_tokens;
  }
  if (release) $('version').textContent = `${release.edition} · v${release.version}`;
  const preview = gpu.device.toLowerCase().includes('mock');
  $('status-label').textContent = preview ? 'Frontend preview' : gpu.is_loaded ? (gpu.has_lora ? 'Trained model ready' : 'Base model loaded') : gpu.adapter_available === false ? 'Adapter missing' : 'Ready to chat';
  $('status-dot').className = gpu.adapter_available === false && !preview ? 'status-dot offline' : 'status-dot online';
  $('health').textContent = `${gpu.device}\n${gpu.is_loaded ? (gpu.shared_memory ? 'Native process RAM: ' + (gpu.native_process_ram_gb ?? 'unavailable') + ' GB' : 'VRAM: ' + gpu.vram_allocated_gb + ' GB') : 'The first reply takes a little longer while the model loads.'}`;
 } catch { $('status-label').textContent = 'Server offline'; $('status-dot').className = 'status-dot offline'; $('health').textContent = 'Start run_trained_chat.bat and refresh this page.'; }
}
$('clear').onclick = () => { if (controller) return; history = []; globalThis.beaconVisitor?.newChat(); $('messages').innerHTML = welcome; $('metrics').textContent = ''; $('error').textContent = ''; $('activity').textContent = ''; $('prompt').value = ''; $('prompt').focus(); };
$('export').onclick = () => {
 if (!history.length) { $('activity').textContent = 'Complete a conversation first, then save it.'; return; }
 const text = 'BEACON v1.0.0\n\n' + history.map(m => `${m.role === 'user' ? 'YOU' : 'BEACON'}\n${m.content}`).join('\n\n');
 const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
 const link = document.createElement('a'); link.href = url; link.download = 'beacon-chat.txt'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('messages').addEventListener('click', event => { const suggestion = event.target.closest('[data-prompt]'); if (suggestion && !controller) { $('prompt').value = suggestion.dataset.prompt; $('prompt').focus(); } });
$('stop').onclick = () => controller?.abort();
$('prompt').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); if (!controller) $('form').requestSubmit(); } });
$('form').onsubmit = async event => {
 event.preventDefault(); if (controller) return;
 const message = $('prompt').value.trim(); if (!message) return;
 if (!$('temperature').checkValidity() || !$('tokens').checkValidity()) { $('error').textContent = 'Check the temperature (0–2) and response token limit (16–2048) in settings.'; return; }
 controller = new AbortController(); $('send').disabled = true; $('clear').disabled = true; $('stop').hidden = false; $('send').hidden = true;
 $('error').textContent = ''; $('metrics').textContent = ''; $('messages').querySelector('.welcome')?.remove(); $('activity').textContent = 'BEACON is preparing a reply…';
 bubble('user', message); const reply = bubble('assistant', ''); reply.card.classList.add('partial'); $('prompt').value = '';
 let finishReason = '', droppedHistory = false;
 try {
  const response = await fetch('/api/chat/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
   body: JSON.stringify({ message, history, use_base_model: $('base').checked, think: $('think').checked, temperature: Number($('temperature').value), max_new_tokens: Number($('tokens').value), system_prompt: $('system').value.trim() || null,
    client_id: globalThis.beaconVisitor?.clientId, session_id: globalThis.beaconVisitor?.sessionId }) });
  if (!response.ok) throw new Error(globalThis.beaconVisitor ? 'BEACON is temporarily unavailable. Please ask a seminar member.' : `Chat failed (${response.status}). Check your settings or the server window.`);
  await readUIMessageStream(response.body, part => {
   if (part.type === 'text-delta') { reply.body.textContent += part.delta; $('activity').textContent = 'BEACON is replying…'; scrollChat(); }
   if (part.type === 'data-metrics') { const m = part.data; droppedHistory = m.history_messages_dropped > 0; $('metrics').textContent = `${m.total_tokens} tokens · ${m.tokens_per_second} tokens/s · ${m.elapsed_seconds}s${m.time_to_first_text_seconds != null ? ' · First text: ' + m.time_to_first_text_seconds + 's' : ''}${m.prompt_tokens != null ? ' · Input: ' + m.prompt_tokens + ' tokens' : ''}`; }
   if (part.type === 'finish') finishReason = part.finishReason;
  });
  if (!reply.body.textContent.trim()) throw new Error('No answer was produced. Try again with Thinking mode off.');
  history.push({ role: 'user', content: message }, { role: 'assistant', content: reply.body.textContent }); reply.card.classList.remove('partial');
  $('activity').textContent = (finishReason === 'length' ? (Number($('tokens').max) <= 192 ? 'Laptop response limit reached. Ask a focused follow-up. ' : 'Response limit reached. Ask a follow-up or increase the token limit. ') : '') + (droppedHistory ? 'BEACON used recent conversation context to keep this reply fast. Earlier messages remain on screen.' : '');
 } catch (error) {
  $('error').textContent = error.name === 'AbortError' ? 'Reply stopped. The partial response was not added to conversation history.' : error.message;
  if (!reply.body.textContent) reply.body.textContent = 'No completed reply.';
  $('prompt').value = message; $('activity').textContent = '';
 } finally { controller = null; $('send').disabled = false; $('clear').disabled = false; $('stop').hidden = true; $('send').hidden = false; health(); $('prompt').focus(); }
};
health();
