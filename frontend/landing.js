/* Presentation only: composer auto-height, the landing demo composer and scroll reveals. Chat logic lives in app.js. */
(function () {
 const prompt = document.getElementById('prompt') || document.getElementById('demo-prompt');
 if (prompt) {
  const fit = () => { prompt.style.height = 'auto'; prompt.style.height = Math.min(prompt.scrollHeight, 140) + 'px'; };
  prompt.addEventListener('input', fit);
  prompt.form.addEventListener('submit', () => setTimeout(fit));
 }

 // The landing card is a demo: Enter opens the chat page with the typed question (see form action).
 const demo = document.getElementById('demo-prompt');
 if (demo) demo.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); demo.form.requestSubmit(); }
 });
 if (demo) demo.form.addEventListener('submit', event => { if (!demo.value.trim()) { event.preventDefault(); location.href = 'chat.html'; } });

 const els = document.querySelectorAll('[data-reveal]');
 if (!els.length || !('IntersectionObserver' in window) || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
 els.forEach(el => { el.style.opacity = 0; el.style.transform = 'translateY(24px)'; el.style.transition = 'opacity 700ms cubic-bezier(.2,.6,.2,1), transform 700ms cubic-bezier(.2,.6,.2,1)'; });
 const io = new IntersectionObserver(entries => entries.forEach(e => {
  if (!e.isIntersecting) return;
  e.target.style.opacity = 1; e.target.style.transform = 'none'; io.unobserve(e.target);
  // Hand hover transitions back to the stylesheet once revealed.
  setTimeout(() => { e.target.style.transition = ''; e.target.style.transform = ''; }, 750);
 }), { threshold: 0.12 });
 els.forEach(el => io.observe(el));
})();
