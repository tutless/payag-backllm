from server import Server
from servicer import sample_servicer
from proto import sample_pb2_grpc


class ServicerFactory:
    def __init__(self, serv: Server):
        self.server = serv

    def services(self) -> None:
        sample_pb2_grpc.add_SampleServiceServicer_to_server(
            sample_servicer.SampleService(), self.server.server
        )
