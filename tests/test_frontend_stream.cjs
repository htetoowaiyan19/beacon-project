const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readUIMessageStream } = require('../frontend/stream.js');

function body(text, chunkSize = 1) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({ start(controller) {
    for (let i = 0; i < bytes.length; i += chunkSize) controller.enqueue(bytes.slice(i, i + chunkSize));
    controller.close();
  }});
}

test('split UTF-8 Burmese and CRLF frames survive transport boundaries', async () => {
  const events = [];
  await readUIMessageStream(body('data: {"type":"text-delta","id":"t","delta":"မင်္ဂလာပါ"}\r\n\r\ndata: {"type":"finish"}\r\n\r\ndata: [DONE]\r\n\r\n'), e => events.push(e));
  assert.equal(events[0].delta, 'မင်္ဂလာပါ');
  assert.equal(events[1].type, 'finish');
});

test('server errors reach the caller', async () => {
  await assert.rejects(readUIMessageStream(body('data: {"type":"error","errorText":"failed"}\n\n'), () => {}), /failed/);
});

test('truncated streams cannot be recorded as completed replies', async () => {
  await assert.rejects(readUIMessageStream(body('data: {"type":"text-delta","id":"t","delta":"hi"}\n\n'), () => {}), /before completion/);
});
