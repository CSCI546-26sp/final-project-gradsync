import grpc
from concurrent import futures
from .proto import cluster_service_pb2
from .proto import cluster_service_pb2_grpc
import psutil


class ClusterServer(cluster_service_pb2_grpc.ClusterCoordinatorServicer):
    """
    gRPC Server Handler. Directly mutates the passed in `node` object's state
    under its own thread-safe lock.
    """
    def __init__(self, node):
        self.node = node

    def Ping(self, request, context):
        """Simplest handler. Just proves the gRPC listener is running."""
        return cluster_service_pb2.Ack(ok=True)

    def RequestVote(self, request, context):
        with self.node._election_cv:
            # If we already finalized the cluster topology, the election is permanently over.
            if self.node.topology_config is not None:
                return cluster_service_pb2.VoteResponse(
                    term=self.node.current_term,
                    vote_granted=False
                )

            # Rule 1: Step down if candidate has a stricter/higher term
            if request.term > self.node.current_term:
                self.node.current_term = request.term
                # Resolving the FOLLOWER enum type dynamically to avoid circular import!
                self.node.state = type(self.node.state).FOLLOWER
                self.node.voted_for = None
                self.node._election_cv.notify_all()
            
            # Rule 2: Grant vote if term matches and we haven't voted for someone else yet
            vote_granted = False
            if request.term == self.node.current_term:
                if self.node.voted_for is None or self.node.voted_for == request.candidate_ip:
                    self.node.state = type(self.node.state).FOLLOWER
                    self.node.voted_for = request.candidate_ip
                    vote_granted = True
                    print(f"[{self.node.host_ip}] Granted vote to {request.candidate_ip} (term {self.node.current_term})")
                    self.node._election_cv.notify_all()
            
            return cluster_service_pb2.VoteResponse(
                term=self.node.current_term,
                vote_granted=vote_granted
            )

    def BroadcastTopology(self, request, context):
        with self.node._election_cv:
            capacity = psutil.virtual_memory().available
            # Record our own capacity in peer_capacities so the Leader sees it when returning from join_cluster
            if self.node.state == type(self.node.state).LEADER:
                self.node.peer_capacities[self.node.host_ip] = capacity
                
            # If we already finalized the cluster topology, ignore duplicates
            if self.node.topology_config is not None:
                return cluster_service_pb2.TopologyResponse(ok=True, available_memory_bytes=capacity)

            # Reject topology if it comes from an older leader
            if request.term < self.node.current_term:
                return cluster_service_pb2.TopologyResponse(ok=False, available_memory_bytes=capacity)

            # If leader has a newer term, update ourselves
            if request.term > self.node.current_term:
                self.node.current_term = request.term
                self.node.state = type(self.node.state).FOLLOWER
                self.node.voted_for = None

            self.node.topology_config = request
            self.node.coordinator_ip = request.coordinator_ip
            self.node.state = type(self.node.state).FOLLOWER
            print(f"[{self.node.host_ip}] Received cluster topology! Coordinator is {self.node.coordinator_ip}")
            # Wake up the main thread waiting in join_cluster()
            self.node._election_cv.notify_all()
            
        return cluster_service_pb2.TopologyResponse(ok=True, available_memory_bytes=capacity)

    def BroadcastPartitioning(self, request, context):
        with self.node._election_cv:
            self.node.partition_config = request
            print(f"[{self.node.host_ip}] Received layer partition boundaries: start={request.start_layer_idx}, end={request.end_layer_idx}")
            self.node._election_cv.notify_all()
        return cluster_service_pb2.Ack(ok=True)

