from langchain_community.chat_message_histories import ChatMessageHistory


class TrimmedChatMessageHistory(ChatMessageHistory):
    def __init__(self, max_msg=3):
        super().__init__()
        self.max_msg = max_msg

    def add_message(self, message):
        super().add_message(message)
        if len(self.messages) > self.max_msg:
            self.messages = self.messages[-self.max_msg :]
