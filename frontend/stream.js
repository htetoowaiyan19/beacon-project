/* Incremental UTF-8 SSE parser for the Vercel AI SDK UI Message Stream. */
(function (root) {
  async function readUIMessageStream(body, onEvent) {
    if (!body) throw new Error("The server returned no stream");
    const reader = body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let terminated = false;
    let finished = false;
    function consume(frame) {
      const data = frame.split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, "")).join("\n");
      if (!data) return;
      if (data === "[DONE]") { terminated = true; return; }
      const event = JSON.parse(data);
      if (event.type === "finish") finished = true;
      if (event.type === "error") throw new Error(event.errorText || "Generation failed");
      onEvent(event);
    }
    try {
      while (!terminated) {
        const { value, done } = await reader.read();
        buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
        let boundary;
        while ((boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
          consume(buffer.slice(0, boundary.index));
          buffer = buffer.slice(boundary.index + boundary[0].length);
        }
        if (done) {
          if (buffer.trim()) consume(buffer);
          break;
        }
      }
      if (!terminated || !finished) throw new Error("Chat stream ended before completion");
    } finally {
      if (!terminated) await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  }
  if (typeof module !== "undefined" && module.exports) module.exports = { readUIMessageStream };
  else root.readUIMessageStream = readUIMessageStream;
})(globalThis);
