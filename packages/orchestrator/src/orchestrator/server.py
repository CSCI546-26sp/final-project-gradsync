import grpc
from concurrent import futures
from .proto import cluster_service_pb2
from .proto import cluster_service_pb2_grpc


class ClusterServer(cluster_service_pb2_grpc.ClusterCoordinatorServicer):
    """
    gRPC Server Handler. Directly mutates the passed in `node` object's state
    under its own thread-safe lock.
    """
    def __init__(self, node):
        self.node = node

    def RequestVote(self, request, context):
        with self.node._election_cv:
            # Rule 1: Step down if candidate has a stricter/higher term
            if request.term > self.node.current_term:
                self.node.current_term = request.term
                # Resolving the FOLLOWER enum type dynamically to avoid circular import!
                self.node.state = type(self.node.state).FOLLOWER
                self.node.voted_for = None
            
            # Rule 2: Grant vote if term matches and we haven't voted for someone else yet
            vote_granted = False
            if request.term == self.node.current_term:
                if self.node.voted_for is None or self.node.voted_for == request.candidate_ip:
                    self.node.state = type(self.node.state).FOLLOWER
                    self.node.voted_for = request.candidate_ip
                    vote_granted = True
                    print(f"[{self.node.host_ip}] Granted vote to {request.candidate_ip} (term {self.node.current_term})")
            
            return cluster_service_pb2.VoteResponse(
                term=self.node.current_term,
                vote_granted=vote_granted
            )

    def BroadcastTopology(self, request, context):
        with self.node._election_cv:
            self.node.topology_config = request
            self.node.coordinator_ip = request.coordinator_ip
            self.node.state = type(self.node.state).FOLLOWER
            print(f"[{self.node.host_ip}] Received cluster topology! Coordinator is {self.node.coordinator_ip}")
            # Wake up the main thread waiting in join_cluster()
            self.node._election_cv.notify_all()
            
        return cluster_service_pb2.Ack(ok=True)


def serve_cluster(node, port=50051):
    """Starts the background gRPC server for the Raft node."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    cluster_service_pb2_grpc.add_ClusterCoordinatorServicer_to_server(
        ClusterServer(node), server
    )
    server.add_insecure_port(f'[::]:{port}')
    print(f"[{node.host_ip}] Raft Server listening on port {port}...")
    server.start()
    return server
