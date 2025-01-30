import logging
import string
from typing import TypeVar
from proto import payag_pb2, payag_pb2_grpc
from concurrent import futures
import grpc
from grpc_reflection.v1alpha import reflection
import reactivex as rx
from servicer.greet_servicer import Greeter

T = TypeVar("T")


class PayagServer:
    def __init__(self):
        self.port = "51150"
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        self.service_name = [("payag.Greeter", reflection.SERVICE_NAME)]

    def reflection(self):
        for service_name in self.service_name:
            reflection.enable_server_reflection(service_name, self.server)

    def start_server(self):
        payag_pb2_grpc.add_GreeterServicer_to_server(Greeter(), self.server)
        self.server.add_insecure_port(f"[::]:{self.port}")
        self.server.start
        print(f"Server listening to port {self.port}")
        self.server.wait_for_termination()


if __name__ == "__main__":
    payag = PayagServer()
    payag.start_server()
    logging.basicConfig(level=logging.DEBUG)
