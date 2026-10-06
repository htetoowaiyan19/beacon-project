const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const code = fs.readFileSync('frontend/show-day.js', 'utf8');

function setup(visitorMode) {
 const calls = [], listeners = {}, intervals = [];
 const notice = { hidden: true }; let count = 0;
 const context = vm.createContext({
  crypto: { randomUUID: () => 'anonymous-' + ++count }, Date,
  document: { body: { classList: { contains: () => visitorMode } },
   getElementById: id => id === 'monitor-notice' ? notice : { addEventListener: (event, fn) => { listeners[id + ':' + event] = fn; } } },
  fetch: async (url, options) => { calls.push({ url, body: JSON.parse(options.body) }); return { ok: true }; },
  setInterval: fn => intervals.push(fn),
 });
 vm.runInContext(code, context);
 return { calls, listeners, intervals, notice, context };
}
test('normal chat emits no visitor telemetry', () => {
 const ui = setup(false); assert.equal(ui.calls.length, 0); assert.equal(ui.notice.hidden, true);
});
test('visitor telemetry signals activity without collecting draft text', () => {
 const ui = setup(true); assert.equal(ui.notice.hidden, false);
 assert.equal(ui.calls[0].body.event, 'connected');
 ui.listeners['prompt:input']({ target: { value: 'private unsent draft' } });
 ui.listeners['prompt:input']({ target: { value: 'another private draft' } });
 assert.equal(ui.calls.filter(c => c.body.event === 'typing').length, 1);
 assert.equal(JSON.stringify(ui.calls).includes('private'), false);
 ui.intervals[0](); assert.equal(ui.calls.at(-1).body.event, 'heartbeat');
 const before = vm.runInContext('beaconVisitor.sessionId', ui.context);
 vm.runInContext('beaconVisitor.newChat()', ui.context);
 assert.notEqual(vm.runInContext('beaconVisitor.sessionId', ui.context), before);
 assert.equal(ui.calls.at(-1).body.event, 'new_chat');
 assert.equal(ui.calls.at(-1).body.client_id, ui.calls[0].body.client_id);
});
