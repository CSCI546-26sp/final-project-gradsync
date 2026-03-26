class ClusterNode:
    def __init__(self, host_ip, peer_ips):
        self.host_ip = host_ip
        self.peer_ips = peer_ips
        self.state = "FOLLOWER" # FOLLOWER, CANDIDATE, LEADER
        self.coordinator_ip = None
        self.topology_config = None

    def start_lifecycle(self, local_hardware_profile):
        self._run_election()
        
        if self.state == "LEADER":
            # Wait for profiles, map topology, distribute configs
            self.topology_config = self._run_coordinator_logic()
        else:
            # Send profile to leader, block until TopologyConfig is received
            self.topology_config = self._report_to_coordinator(local_hardware_profile)
            
        return self.topology_config