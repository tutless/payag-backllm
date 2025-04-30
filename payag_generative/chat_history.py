from langchain_community.chat_message_histories import ChatMessageHistory


class TrimmedChatMessageHistory(ChatMessageHistory):
    def __init__(self, max_messages=6):
        super.__init__()
        self.max_messages = max_messages

    def add_message(self, message):
        super().add_message(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]
