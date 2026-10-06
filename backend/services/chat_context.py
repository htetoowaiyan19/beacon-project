"""Keep recent complete exchanges within a tokenizer-measured inference budget."""


class ChatContextError(ValueError):
    """A safe, actionable input error that can be shown in the chat UI."""


def prepare_chat(tokenizer, messages, system_prompt, *, think=False, max_prompt_tokens=2048,
                 max_history_turns=3, context_limit=None, max_new_tokens=512):
    if not messages or messages[-1].get('role') != 'user':
        raise ChatContextError('A conversation must end with your current user message.')
    budget = min(max_prompt_tokens, context_limit - max_new_tokens) if context_limit else max_prompt_tokens
    # Group each historical question with its answer, so trimming never leaves an orphan answer.
    groups = []
    for message in messages[:-1]:
        if message['role'] == 'user':
            groups.append([message])
        elif groups:
            groups[-1].append(message)
    selected = groups[-max_history_turns:] if max_history_turns > 0 else []

    def encode(history):
        conversation = ([{'role': 'system', 'content': system_prompt}] if system_prompt else [])
        conversation += [message for group in history for message in group] + [messages[-1]]
        text = tokenizer.apply_chat_template(conversation, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=think)
        return tokenizer(text, return_tensors='pt'), conversation

    # Always preserve the current message and instructions verbatim; never cut Burmese byte sequences.
    encoded, conversation = encode([])
    if encoded['input_ids'].shape[-1] > budget:
        raise ChatContextError('Your message and instructions are too long for fast chat. Shorten the message or custom instructions and try again.')
    if selected:
        while True:
            encoded, conversation = encode(selected)
            if encoded['input_ids'].shape[-1] <= budget:
                break
            selected.pop(0)
    retained = sum(len(group) for group in selected)
    return encoded, conversation, dict(history_turns_used=len(selected),
                                      history_messages_dropped=len(messages) - 1 - retained,
                                      prompt_token_budget=budget)
