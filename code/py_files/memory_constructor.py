class BaseMemoryConstrucor:
    def __init__(self, system_message, human_symbol, ai_symbol):
        self.system_message = system_message
        self.human_symbol = human_symbol
        self.ai_symbol = ai_symbol
        self.memory = []  # [(human_input, bot_response)]

    def get_full_conversation(self):
        conversation = self.system_message
        for human, ai in self.memory:
            conversation += (
                f"{self.human_symbol} {human}\n" + f"{self.ai_symbol} {ai}\n"
            )
        return conversation

    def clear_memory(self):
        self.memory = []

    def add_to_memory(self, human_input, ai_response):
        self.memory.append((human_input, ai_response))

    def pop_from_memory(self):
        if len(self.memory) > 0:
            self.memory.pop()

    def get_used_memory(self):
        pass

    def construct_input_memory(self, human_input):
        return (
            self.get_used_memory()
            + f"{self.human_symbol} {human_input}\n"
            + f"{self.ai_symbol} "
        )


class FullMemoryConstructor(BaseMemoryConstrucor):
    def get_used_memory(self):
        return self.get_full_conversation()


class FixedWindowLengthMemoryConstructor(BaseMemoryConstrucor):
    def __init__(self, window_length, system_message, human_symbol, ai_symbol):
        super().__init__(system_message, human_symbol, ai_symbol)
        self.window_length = window_length

    def get_used_memory(self):
        conversation = self.system_message
        for human, ai in self.memory[-self.window_length :]:
            conversation += (
                f"{self.human_symbol} {human}\n" + f"{self.ai_symbol} {ai}\n"
            )
        return conversation


class DynamicWindowLengthMemoryConstructor(BaseMemoryConstrucor):
    def __init__(self, tokenizer, max_tokens, system_message, human_symbol, ai_symbol):
        super().__init__(system_message, human_symbol, ai_symbol)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def get_used_memory(self):
        total_tokens = len(self.tokenizer.tokenize(self.system_message))
        window_length = 0
        for human, ai in self.memory[::-1]:
            message = f"{self.human_symbol} {human}\n" + f"{self.ai_symbol} {ai}\n"
            total_tokens += len(self.tokenizer.tokenize(message))
            if total_tokens > self.max_tokens:
                break
            window_length += 1

        conversation = self.system_message
        for human, ai in self.memory[-window_length:]:
            conversation += (
                f"{self.human_symbol} {human}\n" + f"{self.ai_symbol} {ai}\n"
            )
        return conversation
