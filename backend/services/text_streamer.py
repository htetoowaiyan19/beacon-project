"""Display decoded Unicode promptly, including Burmese text without spaces."""
from transformers import TextIteratorStreamer


class UnicodeTextIteratorStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, **kwargs):
        kwargs['clean_up_tokenization_spaces'] = False
        super().__init__(tokenizer, **kwargs)

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError('Text streaming supports one conversation at a time')
        if len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        # Byte-level tokens may contain only part of a UTF-8 character. Wait for
        # those final bytes, but do not wait for a space or a complete sentence.
        stable = text.rstrip('\ufffd')
        printable = stable[self.print_len:]
        self.print_len = len(stable)
        if printable:
            self.on_finalized_text(printable)
