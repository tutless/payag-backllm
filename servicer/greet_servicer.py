from proto import payag_pb2_grpc, payag_pb2


class Greeter(payag_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        print("SayHello method called")
        print("SayHello method called")
        return payag_pb2.HelloReply(message=f"Hello {request.name}")
