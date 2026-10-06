const $ = id => document.getElementById(id);
let history = [], controller = null, maxTokens = 512;
const welcome = $('messages').innerHTML;
function scrollChat() { $('messages').scrollTop = $('messages').scrollHeight; }
function el(tag, className) { const node = document.createElement(tag); node.className = className; return node; }
function bubble(role, text) {
 if (role === 'user') { const card = el('div', 'message user'); card.textContent = text; $('messages').appendChild(card); scrollChat(); return { card, body: card }; }
 const card = el('div', 'message assistant'), avatar = el('div', 'avatar'), logo = document.createElement('img');
 logo.src = 'beacon-logo.png'; logo.alt = ''; avatar.appendChild(logo);
 const body = el('div', 'message-body'); body.textContent = text;
 card.append(avatar, body); $('messages').appendChild(card); scrollChat(); return { card, body };
}
function dots() { const wrap = el('span', 'dots'); for (let i = 0; i < 3; i++) wrap.appendChild(document.createElement('span')); return wrap; }
async function health() {
 try {
  const response = await fetch('/api/health'); if (!response.ok) throw new Error('Health check failed');
  const { gpu, release } = await response.json();
  if (gpu.runtime === 'llama.cpp' && gpu.max_new_tokens) maxTokens = Math.min(512, gpu.max_new_tokens);
  if (release) $('version').textContent = 'v' + release.version;
  const preview = gpu.device.toLowerCase().includes('mock');
  $('status-label').textContent = preview ? 'Frontend preview' : gpu.is_loaded ? (gpu.has_lora ? 'Trained model ready' : 'Base model loaded') : gpu.adapter_available === false ? 'Adapter missing' : 'Ready to chat';
  $('status-dot').className = gpu.adapter_available === false && !preview ? 'status-dot offline' : 'status-dot online';
  $('health').textContent = gpu.device;
 } catch {
  $('status-label').textContent = 'Server offline'; $('status-dot').className = 'status-dot offline';
  if (!$('error').textContent) $('error').textContent = 'B.E.A.C.O.N. is offline. Start the server and refresh this page.';
 }
}
$('clear').onclick = () => { if (controller) return; history = []; globalThis.beaconVisitor?.newChat(); $('messages').innerHTML = welcome; $('metrics').textContent = ''; $('error').textContent = ''; $('activity').textContent = ''; $('prompt').value = ''; $('prompt').focus(); };
$('messages').addEventListener('click', event => { const suggestion = event.target.closest('[data-prompt]'); if (suggestion && !controller) { $('prompt').value = suggestion.dataset.prompt; $('form').requestSubmit(); } });
$('stop').onclick = () => controller?.abort();
$('prompt').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); if (!controller) $('form').requestSubmit(); } });
$('form').onsubmit = async event => {
 event.preventDefault(); if (controller) return;
 const message = $('prompt').value.trim(); if (!message) return;
 controller = new AbortController(); $('send').disabled = true; $('clear').disabled = true; $('stop').hidden = false; $('send').hidden = true;
 $('error').textContent = ''; $('metrics').textContent = ''; $('messages').querySelector('.welcome')?.remove(); $('activity').textContent = '';
 bubble('user', message); const reply = bubble('assistant', ''); reply.body.appendChild(dots()); $('prompt').value = '';
 let finishReason = '', droppedHistory = false;
 try {
  const response = await fetch('/api/chat/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
   body: JSON.stringify({ message, history, use_base_model: false, think: false, temperature: 0.7, max_new_tokens: maxTokens, system_prompt: null,
    client_id: globalThis.beaconVisitor?.clientId, session_id: globalThis.beaconVisitor?.sessionId }) });
  if (!response.ok) throw new Error(globalThis.beaconVisitor ? 'B.E.A.C.O.N. is temporarily unavailable. Please ask a seminar member.' : `Chat failed (${response.status}). Check the server window.`);
  await readUIMessageStream(response.body, part => {
   if (part.type === 'text-delta') { reply.body.textContent += part.delta; reply.card.classList.add('typing'); scrollChat(); }
   if (part.type === 'data-metrics') { const m = part.data; droppedHistory = m.history_messages_dropped > 0; $('metrics').textContent = `${m.total_tokens} tokens · ${m.tokens_per_second} tokens/s · ${m.elapsed_seconds}s${m.time_to_first_text_seconds != null ? ' · First text: ' + m.time_to_first_text_seconds + 's' : ''}${m.prompt_tokens != null ? ' · Input: ' + m.prompt_tokens + ' tokens' : ''}`; }
   if (part.type === 'finish') finishReason = part.finishReason;
  });
  if (!reply.body.textContent.trim()) throw new Error('No answer was produced. Please try again.');
  history.push({ role: 'user', content: message }, { role: 'assistant', content: reply.body.textContent });
  $('activity').textContent = (finishReason === 'length' ? 'Response limit reached. Ask a focused follow-up. ' : '') + (droppedHistory ? 'B.E.A.C.O.N. used recent conversation context to keep this reply fast. Earlier messages remain on screen.' : '');
 } catch (error) {
  $('error').textContent = error.name === 'AbortError' ? 'Reply stopped. The partial response was not added to conversation history.' : error.message;
  if (!reply.body.textContent) reply.body.textContent = 'No completed reply.';
  $('prompt').value = message; $('activity').textContent = '';
 } finally { reply.card.classList.remove('typing'); controller = null; $('send').disabled = false; $('clear').disabled = false; $('stop').hidden = true; $('send').hidden = false; health(); $('prompt').focus(); }
};
health();
// A question chosen on the landing page arrives as ?q= and is sent once.
const initial = typeof location !== 'undefined' ? new URLSearchParams(location.search).get('q') : null;
if (initial?.trim()) { window.history.replaceState(null, '', location.pathname); $('prompt').value = initial; $('form').requestSubmit(); }
