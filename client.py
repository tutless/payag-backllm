import grpc
from proto import payag_pb2, payag_pb2_grpc


def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = payag_pb2_grpc.GreeterStub(channel)
        response = stub.SayHello(payag_pb2.HelloRequest(name="Kenneth"))
        print("Greeter client received: " + response.message)


if __name__ == "__main__":
    run()
