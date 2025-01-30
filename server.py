from concurrent import futures
import grpc
from proto import payag_pb2_grpc
import logging
from grpc_reflection.v1alpha import reflection
from servicer.payag_service import PayagService


def serve():
    port = "51150"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    payag_pb2_grpc.add_PayagServiceServicer_to_server(PayagService(), server)

    # implement reflection
    rf = reflection.SERVICE_NAME
    service_name = ("payag.Greeter", rf)
    reflection.enable_server_reflection(service_name, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Server start listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":

    serve()
    logging.basicConfig(level=logging.DEBUG)
