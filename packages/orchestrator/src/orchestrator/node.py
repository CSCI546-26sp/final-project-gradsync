import random
import threading
import time
from enum import Enum, auto
import grpc
from concurrent.futures import ThreadPoolExecutor, as_completed

from .client import ClusterClient
from .server import ClusterServer
from .proto import cluster_service_pb2, cluster_service_pb2_grpc


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
        self.server = None
        self._election_cv = threading.Condition()  # used to sleep until timeout (or future heartbeat)

    def _serve_cluster(self):
        """Starts the background gRPC server for the Raft node."""
        server = grpc.server(ThreadPoolExecutor(max_workers=10))
        cluster_service_pb2_grpc.add_ClusterCoordinatorServicer_to_server(
            ClusterServer(self), server
        )
        server.add_insecure_port(f'{self.host_ip}:{self.port}')
        print(f"[{self.host_ip}] Raft Server listening on {self.host_ip}:{self.port}...")
        server.start()
        return server

    def join_cluster(self):
        """
        Main orchestrator lifecycle. Blocks until the cluster agrees on a single
        coordinator (LEADER) and a finalized network topology.
        Returns the finalized topology (TopologyConfig).
        """
        # Start the background gRPC server to receive votes and topology
        self.server = self._serve_cluster()
        
        # Halt execution and wait for all pipeline peers to come online completely
        self.wait_for_peers()

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

                # Immediately check if majority is met (handles 1-node cluster zero-peer case)
                total_nodes = len(self.peer_ips) + 1
                if self.votes_received > total_nodes // 2:
                    self.state = NodeState.LEADER
                    print(f"[{self.host_ip}] Elected LEADER for term {self.current_term}!")

            # --- LOCK RELEASED ---
            
            if self.state == NodeState.LEADER:
                self.broadcast_topology()
                continue

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
                            is_leader_now = False
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
                                    is_leader_now = True
                            
                            if is_leader_now:
                                self.broadcast_topology()
                                break

                    except Exception as e:
                        peer_ip = futures[future]
                        print(f"[{self.host_ip}] Failed to get vote from {peer_ip}. Re-trying next cycle.")
        
        return self.topology_config

    def shutdown(self):
        """Cleanly stop the gRPC server."""
        if self.server:
            self.server.stop(grace=None)

    def wait_for_peers(self):
        """Blocks indefinitely until all peer IPs return a successful gRPC Ping."""
        if not self.peer_ips:
            return

        print(f"[{self.host_ip}] Waiting for peers to come online: {self.peer_ips}")
        pending = set(self.peer_ips)
        
        while pending:
            # We iterate over a copy (list) so we can safely remove from the original set
            for peer in list(pending):
                client = ClusterClient(target_ip=peer, port=self.port)
                try:
                    if client.ping():
                        print(f"[{self.host_ip}] Peer {peer} is ONLINE!")
                        pending.remove(peer)
                    else:
                        print(f"[{self.host_ip}] Peer {peer} ping returned False.")
                except Exception as e:
                    print(f"[{self.host_ip}] Exception pinging {peer}: {e}")
                finally:
                    client.close()
                    
            if pending:
                time.sleep(0.5)

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
        successes = 1  # We implicitly accept our own topology
        total_nodes = len(self.peer_ips) + 1

        def check_majority():
            # Eagerly lock our topology under mutex to prevent race conditions
            with self._election_cv:
                if successes > total_nodes // 2 and self.topology_config is None:
                    topology = cluster_service_pb2.TopologyConfig(
                        coordinator_ip=self.host_ip,
                        ordered_node_ips=ordered_ips,
                        term=self.current_term
                    )
                    self.topology_config = topology
                    self.coordinator_ip = self.host_ip
                    self._election_cv.notify_all()

        # Check immediately for single-node clusters before engaging executor
        check_majority()

        # Broadcast to all peers concurrently
        with ThreadPoolExecutor(max_workers=max(1, len(self.peer_ips))) as executor:
            futures = [
                executor.submit(self.send_topology, peer, self.host_ip, ordered_ips, self.current_term)
                for peer in self.peer_ips
            ]
            
            for future in as_completed(futures):
                try:
                    if future.result():
                        successes += 1
                        check_majority()
                except Exception as e:
                    pass

        # If the executor finishes and we STILL haven't reached majority, we failed.
        with self._election_cv:
            if self.topology_config is None:
                self.state = NodeState.FOLLOWER
                self.voted_for = None

    def send_topology(self, peer_ip: str, coordinator_ip: str, ordered_node_ips: list, term: int) -> bool:
        """Helper to send the topology via the client layer."""
        client = ClusterClient(target_ip=peer_ip, port=self.port)
        try:
            return client.broadcast_topology(coordinator_ip, ordered_node_ips, term)
        finally:
            client.close()