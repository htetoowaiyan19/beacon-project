/* Seminar visitor activity: submitted chats are observed server-side, never drafts. */
(function () {
 if (!document.body.classList.contains('visitor-mode')) return;
 const id = () => crypto.randomUUID();
 const visitor = { clientId: id(), sessionId: id() };
 let lastTyping = 0;
 function signal(event) {
  fetch('/api/visitor/event', { method: 'POST', headers: { 'Content-Type': 'application/json' },
   body: JSON.stringify({ event, client_id: visitor.clientId, session_id: visitor.sessionId }) }).catch(() => {});
 }
 visitor.newChat = () => { visitor.sessionId = id(); signal('new_chat'); };
 globalThis.beaconVisitor = visitor;
 document.getElementById('monitor-notice').hidden = false;
 document.getElementById('prompt').addEventListener('focus', () => signal('input_focused'));
 document.getElementById('prompt').addEventListener('input', () => {
  if (Date.now() - lastTyping > 5000) { lastTyping = Date.now(); signal('typing'); }
 });
 document.getElementById('stop').addEventListener('click', () => signal('stop'));
 document.getElementById('export').addEventListener('click', () => signal('save'));
 document.getElementById('messages').addEventListener('click', event => { if (event.target.closest('.copy')) signal('copy'); });
 signal('connected'); setInterval(() => signal('heartbeat'), 15000);
})();
