from proto import sample_pb2_grpc, sample_pb2


class SampleServiceServicer(sample_pb2_grpc.SampleServiceServicer):
    def SampleQuery(self, request, context):
        response = sample_pb2.ResAnswer()
        response.answer = f"this response is coming from server {request}"
        return response
