from proto import payag_pb2, payag_pb2_grpc
from payag_llm.llm_core import LLMCore
from payag_generative.generative_core import GenerativeCore


class PayagService(payag_pb2_grpc.PayagServiceServicer):
    def SayHello(self, request, context):
        print("SayHello method called")
        return payag_pb2.HelloReply(message=f"Hello {request.name}")

    def Chat(self, request, context):
        print("Chat method called")
        return payag_pb2.ChatResponse(answer=GenerativeCore.answer(request.query))
        # return payag_pb2.ChatResponse(answer=LLMCore.answer(request.query))
