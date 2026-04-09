import random
import threading
import time
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed

from .client import ClusterClient
from .server import serve_cluster
from .proto import cluster_service_pb2


class NodeState(Enum):
    FOLLOWER  = auto()
    CANDIDATE = auto()
    LEADER    = auto()


class ClusterNode:
    def __init__(self, host_ip: str, peer_ips: list, port: int = 50051):
        self.host_ip = host_ip
        self.peer_ips = peer_ips
        self.port = port
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for = None    # candidate_ip this node voted for in current term
        self.votes_received = 0
        self.coordinator_ip = None
        self.topology_config = None
        self._election_cv = threading.Condition()  # used to sleep until timeout (or future heartbeat)

    def join_cluster(self):
        """
        Blocking call. Runs Raft leader election and returns a TopologyConfig
        once all nodes have agreed on a coordinator.
        """
        # Start the background gRPC server to receive votes and topology
        self.server = serve_cluster(self, self.port)

        while self.topology_config is None:
            # Randomized election timeout: 300–450 ms
            election_timeout = random.uniform(0.300, 0.450)

            with self._election_cv:
                notified = self._election_cv.wait(timeout=election_timeout)
                
                if notified or self.state == NodeState.LEADER:
                    # Woken up by incoming BroadcastTopology (which sets topology_config),
                    # or we somehow are already leader. Loop condition will handle exiting.
                    continue

                # Timeout fired. Transition to CANDIDATE and start election
                self.state = NodeState.CANDIDATE
                self.current_term += 1
                print(f"[{self.host_ip}] Election timeout fired (term={self.current_term}). Becoming CANDIDATE.")

                self.votes_received = 1
                self.voted_for = self.host_ip

                proposed_term = self.current_term
                candidate_for_vote = self.host_ip

            # --- LOCK RELEASED ---
            # Do not hold the lock while making blocking network calls.
            
            if self.state != NodeState.CANDIDATE:
                continue
            print(f"[{self.host_ip}] Requesting votes from peers: {self.peer_ips}")
            # Execute vote requests concurrently
            with ThreadPoolExecutor(max_workers=max(1, len(self.peer_ips))) as executor:
                futures = {
                    executor.submit(self.send_request_vote, peer, proposed_term, candidate_for_vote): peer
                    for peer in self.peer_ips
                }

                for future in as_completed(futures):
                    try:
                        vote_granted = future.result()
                        
                        if vote_granted:
                            # --- ACQUIRE LOCK BRIEFLY ---
                            with self._election_cv:
                                # If another peer won the election or we started a new term, ignore stale votes
                                if self.state != NodeState.CANDIDATE or self.current_term != proposed_term:
                                    break
                                    
                                self.votes_received += 1
                                
                                # Check majority (N/2 + 1 where N = self + peers)
                                total_nodes = len(self.peer_ips) + 1
                                if self.votes_received > total_nodes // 2:
                                    self.state = NodeState.LEADER
                                    print(f"[{self.host_ip}] Elected LEADER for term {self.current_term}!")
                                    self.broadcast_topology()
                                    break
                    except Exception as e:
                        peer_ip = futures[future]
                        print(f"[{self.host_ip}] Failed to get vote from {peer_ip}. Re-trying next cycle.")
        
        # Stop background server gracefully once election ends
        self.server.stop(grace=None)
        
        return self.topology_config

    def send_request_vote(self, peer_ip: str, term: int, candidate_ip: str) -> bool:
        """Sends RequestVote gRPC to peer_ip."""
        client = ClusterClient(target_ip=peer_ip, port=self.port)
        try:
            vote_granted, responder_term = client.request_vote(term, candidate_ip)

            # Standard Raft rule: If a peer responds with a term greater than ours, 
            # we are out-of-date and must immediately revert to FOLLOWER
            with self._election_cv:
                if responder_term > self.current_term:
                    self.current_term = responder_term
                    self.state = NodeState.FOLLOWER
                    self.voted_for = None
                    print(f"[{self.host_ip}] Peer {peer_ip} has higher term {responder_term}. Stepping down to FOLLOWER.")

            return vote_granted
        finally:
            client.close()

    def broadcast_topology(self):
        """Called by the newly elected LEADER to tell all peers the final assignments."""
        ordered_ips = [self.host_ip] + self.peer_ips
        topology = cluster_service_pb2.TopologyConfig(
            coordinator_ip=self.host_ip,
            ordered_node_ips=ordered_ips
        )
        # Update our own local state so our `while` loop exits
        self.topology_config = topology
        self.coordinator_ip = self.host_ip

        print(f"[{self.host_ip}] Broadcasting topology to peers...")
        # Broadcast to all peers concurrently
        with ThreadPoolExecutor(max_workers=max(1, len(self.peer_ips))) as executor:
            for peer in self.peer_ips:
                executor.submit(self.send_topology, peer, self.host_ip, ordered_ips)

    def send_topology(self, peer_ip: str, coordinator_ip: str, ordered_node_ips: list):
        """Helper to send the topology via the client layer."""
        client = ClusterClient(target_ip=peer_ip, port=self.port)
        try:
            client.broadcast_topology(coordinator_ip, ordered_node_ips)
        finally:
            client.close()


    def start_lifecycle(self, local_hardware_profile):
        self._run_election()

        if self.state == NodeState.LEADER:
            # Wait for profiles, map topology, distribute configs
            self.topology_config = self._run_coordinator_logic()
        else:
            # Send profile to leader, block until TopologyConfig is received
            self.topology_config = self._report_to_coordinator(local_hardware_profile)

        return self.topology_config