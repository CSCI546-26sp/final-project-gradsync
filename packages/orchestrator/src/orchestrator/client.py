import grpc
from .proto import cluster_service_pb2
from .proto import cluster_service_pb2_grpc


class ClusterClient:
    def __init__(self, target_ip="localhost", port=50051):
        self.target_ip = target_ip
        self.port = port
        self.channel = grpc.insecure_channel(f"{target_ip}:{port}")
        self.stub = cluster_service_pb2_grpc.ClusterCoordinatorStub(self.channel)

    def request_vote(self, term: int, candidate_ip: str) -> tuple[bool, int]:
        """
        Sends RequestVote gRPC to the target peer.
        Returns a tuple of (vote_granted: bool, responder_term: int).
        """
        request = cluster_service_pb2.VoteRequest(
            term=term,
            candidate_ip=candidate_ip
        )
        try:
            # 200ms timeout for network call 
            response = self.stub.RequestVote(request, timeout=0.2)
            return response.vote_granted, response.term
        except grpc.RpcError:
            # If the node is unreachable or offline, default to denying vote
            return False, 0

    def broadcast_topology(self, coordinator_ip: str, ordered_node_ips: list, term: int) -> bool:
        """
        Sends BroadcastTopology gRPC to the target peer.
        """
        topology = cluster_service_pb2.TopologyConfig(
            coordinator_ip=coordinator_ip,
            ordered_node_ips=ordered_node_ips,
            term=term
        )
        try:
            # 1.0s timeout for topology dissemination
            response = self.stub.BroadcastTopology(topology, timeout=1.0)
            return response.ok
        except grpc.RpcError:
            return False

    def close(self):
        """Cleanly shut down the gRPC channel."""
        self.channel.close()
