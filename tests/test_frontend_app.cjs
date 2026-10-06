// Exercise chat state and cancellation with a lightweight DOM, without a GPU.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { readUIMessageStream } = require('../frontend/stream.js');

function setup(chatFetch) {
  class Element {
    constructor() { this.value = ''; this.textContent = ''; this.children = []; this.listeners = {}; this.checked = false; this.classList = { add() {}, remove() {} }; }
    appendChild(child) { this.children.push(child); }
    append(...children) { this.children.push(...children); }
    setAttribute() {}
    addEventListener(name, handler) { this.listeners[name] = handler; }
    querySelector() { return null; }
    focus() {}
    checkValidity() { return true; }
  }
  const ids = ['messages','version','status-label','status-dot','health','metrics','clear','export','activity','prompt','error','stop','send','form','base','think','temperature','tokens','system'];
  const elements = Object.fromEntries(ids.map(id => [id, new Element()]));
  elements.temperature.value = '0.7'; elements.tokens.value = '512';
  let submissions = 0;
  elements.form.requestSubmit = () => { submissions++; };
  const context = vm.createContext({
    document: { getElementById: id => elements[id], createElement: () => new Element() },
    fetch: async (url, options) => url === '/api/health'
      ? { ok: true, json: async () => ({ gpu: { device: 'mock', is_loaded: false } }) }
      : chatFetch(options),
    AbortController, Blob, URL, setTimeout, readUIMessageStream,
  });
  vm.runInContext(fs.readFileSync('frontend/app.js', 'utf8'), context);
  return { elements, context, history: () => JSON.parse(vm.runInContext('JSON.stringify(history)', context)), submissions: () => submissions };
}
function streamResponse(complete = true) {
  let text = 'data: {"type":"text-delta","delta":"Hello <script>"}\n\n';
  if (complete) text += 'data: {"type":"finish","finishReason":"stop"}\n\ndata: [DONE]\n\n';
  return { ok: true, body: new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode(text)); c.close(); } }) };
}
const submit = ui => ui.elements.form.onsubmit({ preventDefault() {} });

test('completed replies enter history as text; New conversation clears it', async () => {
  const ui = setup(async () => streamResponse());
  ui.elements.prompt.value = 'Hi'; await submit(ui);
  assert.deepEqual(ui.history(), [{ role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello <script>' }]);
  assert.equal(ui.elements.messages.children[1].children[1].textContent, 'Hello <script>');
  assert.equal(ui.elements.send.disabled, false);
  ui.elements.clear.onclick(); assert.deepEqual(ui.history(), []);
});
test('truncated replies never enter history and restore the prompt', async () => {
  const ui = setup(async () => streamResponse(false));
  ui.elements.prompt.value = 'Retry me'; await submit(ui);
  assert.deepEqual(ui.history(), []);
  assert.equal(ui.elements.prompt.value, 'Retry me');
  assert.match(ui.elements.error.textContent, /before completion/);
});
test('Stop aborts the active request and leaves history unchanged', async () => {
  const ui = setup(options => new Promise((resolve, reject) => options.signal.addEventListener('abort', () => reject(Object.assign(new Error(), { name: 'AbortError' })))));
  ui.elements.prompt.value = 'Stop me'; const pending = submit(ui);
  assert.equal(ui.elements.clear.disabled, true); ui.elements.stop.onclick(); await pending;
  assert.deepEqual(ui.history(), []); assert.equal(ui.elements.prompt.value, 'Stop me');
  assert.equal(ui.elements.stop.hidden, true); assert.equal(ui.elements.send.disabled, false);
});
test('Enter respects Burmese IME composition and Shift+Enter', () => {
  const ui = setup(); const key = ui.elements.prompt.listeners.keydown;
  let prevented = 0; const event = { key: 'Enter', preventDefault() { prevented++; } };
  key({ ...event, isComposing: true }); key({ ...event, keyCode: 229 }); key({ ...event, shiftKey: true });
  assert.equal(ui.submissions(), 0); assert.equal(prevented, 0);
  key(event); assert.equal(ui.submissions(), 1); assert.equal(prevented, 1);
});
