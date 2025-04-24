"""
Server model for queueing systems.

This module implements servers with states and dynamics for use in queueing simulations.
"""

from enum import Enum
from typing import Dict, Optional, Tuple, List, Any, Union, Callable
import numpy as np

class ServerState(Enum):
    """Enumeration of possible server states."""
    IDLE = "idle"  # Server is idle (no customer)
    BUSY = "busy"  # Server is busy serving a customer
    BREAK = "break"  # Server is on a break
    SETUP = "setup"  # Server is in setup for next service
    FAILURE = "failure"  # Server has failed and needs repair
    REPAIR = "repair"  # Server is being repaired
    UNAVAILABLE = "unavailable"  # Server is unavailable for some other reason

class Server:
    """
    A server model for queueing systems with state dynamics.
    
    This class represents a server that can serve customers at a given rate, but can
    also go on breaks, fail, require setup time, etc.
    """
    
    def __init__(
        self,
        id: int,
        service_rate: float,
        state: ServerState = ServerState.IDLE,
        break_rate: float = 0.0,
        break_duration: float = 0.0,
        failure_rate: float = 0.0,
        repair_time: float = 0.0,
        setup_time: float = 0.0,
        state_history: bool = False
    ):
        """
        Initialize a server.
        
        Args:
            id: Unique identifier for the server
            service_rate: Rate at which the server processes customers
            state: Initial state of the server
            break_rate: Rate at which the server goes on breaks when busy
            break_duration: Average duration of a break
            failure_rate: Rate at which the server fails when busy
            repair_time: Average time to repair a failed server
            setup_time: Setup time needed between customers
            state_history: Whether to keep a history of state transitions
        """
        self.id = id
        self.service_rate = service_rate
        self.state = state
        self.break_rate = break_rate
        self.break_duration = break_duration
        self.failure_rate = failure_rate
        self.repair_time = repair_time
        self.setup_time = setup_time
        
        # Statistics
        self.customer_id = None  # ID of the customer being served, if any
        self.current_service_start_time = None  # When the current service started
        self.total_busy_time = 0.0
        self.total_idle_time = 0.0
        self.total_break_time = 0.0
        self.total_repair_time = 0.0
        self.total_setup_time = 0.0
        self.num_services_completed = 0
        self.num_breaks = 0
        self.num_failures = 0
        
        # For state history (optional)
        self.state_history = [] if state_history else None
        self.current_time = 0.0
        self._record_state_if_enabled()
    
    def _record_state_if_enabled(self):
        """Record the current state if state history is enabled."""
        if self.state_history is not None:
            self.state_history.append((self.current_time, self.state))
    
    def start_service(self, customer_id: int, current_time: float) -> bool:
        """
        Start serving a customer.
        
        Args:
            customer_id: ID of the customer to serve
            current_time: Current simulation time
            
        Returns:
            Whether the service was successfully started
        """
        if self.state != ServerState.IDLE and self.state != ServerState.SETUP:
            return False  # Cannot start service if not idle or in setup
        
        self.customer_id = customer_id
        self.current_service_start_time = current_time
        self.state = ServerState.BUSY
        
        self.current_time = current_time
        self._record_state_if_enabled()
        
        return True
    
    def complete_service(self, current_time: float) -> Optional[int]:
        """
        Complete service for the current customer.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            ID of the customer who completed service, or None if no customer was being served
        """
        if self.state != ServerState.BUSY:
            return None  # Cannot complete service if not busy
        
        completed_customer_id = self.customer_id
        service_duration = current_time - self.current_service_start_time
        
        self.total_busy_time += service_duration
        self.num_services_completed += 1
        self.customer_id = None
        self.current_service_start_time = None
        
        # If setup time is required, go to setup state
        if self.setup_time > 0:
            self.state = ServerState.SETUP
        else:
            self.state = ServerState.IDLE
        
        self.current_time = current_time
        self._record_state_if_enabled()
        
        return completed_customer_id
    
    def start_break(self, current_time: float) -> bool:
        """
        The server goes on a break.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Whether the break was successfully started
        """
        # Can only go on break if idle (not serving a customer)
        if self.state != ServerState.IDLE:
            return False
        
        self.state = ServerState.BREAK
        self.num_breaks += 1
        
        self.current_time = current_time
        self._record_state_if_enabled()
        
        return True
    
    def end_break(self, current_time: float) -> bool:
        """
        The server returns from a break.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Whether the break was successfully ended
        """
        if self.state != ServerState.BREAK:
            return False
        
        self.state = ServerState.IDLE
        
        self.current_time = current_time
        self._record_state_if_enabled()
        
        return True
    
    def fail(self, current_time: float) -> bool:
        """
        The server fails and needs repair.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Whether the failure was successfully recorded
        """
        if self.state == ServerState.FAILURE or self.state == ServerState.REPAIR:
            return False  # Already failed or under repair
        
        # Remember the previous state to return to after repair
        self.previous_state = self.state
        
        # Handle failure during service
        if self.state == ServerState.BUSY:
            # Customer's service is interrupted
            service_duration = current_time - self.current_service_start_time
            self.total_busy_time += service_duration
            # Customer will need to be reassigned by the queue system
            self.customer_id = None
            self.current_service_start_time = None
        
        self.state = ServerState.FAILURE
        self.num_failures += 1
        
        self.current_time = current_time
        self._record_state_if_enabled()
        
        return True
    
    def start_repair(self, current_time: float) -> bool:
        """
        Start repairing the server.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Whether the repair was successfully started
        """
        if self.state != ServerState.FAILURE:
            return False  # Can only repair a failed server
        
        self.state = ServerState.REPAIR
        
        self.current_time = current_time
        self._record_state_if_enabled()
        
        return True
    
    def complete_repair(self, current_time: float) -> bool:
        """
        Complete repair of the server.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Whether the repair was successfully completed
        """
        if self.state != ServerState.REPAIR:
            return False  # Not under repair
        
        # After repair, return to idle state
        self.state = ServerState.IDLE
        
        self.current_time = current_time
        self._record_state_if_enabled()
        
        return True
    
    def complete_setup(self, current_time: float) -> bool:
        """
        Complete setup process.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Whether the setup was successfully completed
        """
        if self.state != ServerState.SETUP:
            return False  # Not in setup
        
        self.state = ServerState.IDLE
        
        self.current_time = current_time
        self._record_state_if_enabled()
        
        return True
    
    def set_unavailable(self, current_time: float) -> bool:
        """
        Set the server as unavailable.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Whether the server was successfully set as unavailable
        """
        if self.state == ServerState.BUSY:
            # Cannot make busy server unavailable without handling the customer
            return False
        
        self.state = ServerState.UNAVAILABLE
        
        self.current_time = current_time
        self._record_state_if_enabled()
        
        return True
    
    def set_available(self, current_time: float) -> bool:
        """
        Make the server available again.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Whether the server was successfully made available
        """
        if self.state != ServerState.UNAVAILABLE:
            return False
        
        self.state = ServerState.IDLE
        
        self.current_time = current_time
        self._record_state_if_enabled()
        
        return True
    
    def update_time(self, current_time: float):
        """
        Update the server's current time and accumulators.
        
        Args:
            current_time: Current simulation time
        """
        time_diff = current_time - self.current_time
        
        # Update time accumulators based on current state
        if self.state == ServerState.IDLE:
            self.total_idle_time += time_diff
        elif self.state == ServerState.BUSY:
            self.total_busy_time += time_diff
        elif self.state == ServerState.BREAK:
            self.total_break_time += time_diff
        elif self.state == ServerState.REPAIR:
            self.total_repair_time += time_diff
        elif self.state == ServerState.SETUP:
            self.total_setup_time += time_diff
        
        self.current_time = current_time
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics of the server.
        
        Returns:
            Dictionary with metrics such as utilization, availability, etc.
        """
        total_time = (
            self.total_idle_time + 
            self.total_busy_time + 
            self.total_break_time + 
            self.total_repair_time + 
            self.total_setup_time
        )
        
        if total_time <= 0:
            # Avoid division by zero if no time has passed
            return {
                'state': self.state.value,
                'utilization': 0.0,
                'availability': 1.0,
                'num_services_completed': self.num_services_completed,
                'num_breaks': self.num_breaks,
                'num_failures': self.num_failures
            }
        
        return {
            'state': self.state.value,
            'utilization': self.total_busy_time / total_time,
            'availability': (self.total_idle_time + self.total_busy_time) / total_time,
            'idle_time_fraction': self.total_idle_time / total_time,
            'busy_time_fraction': self.total_busy_time / total_time,
            'break_time_fraction': self.total_break_time / total_time,
            'repair_time_fraction': self.total_repair_time / total_time,
            'setup_time_fraction': self.total_setup_time / total_time,
            'num_services_completed': self.num_services_completed,
            'num_breaks': self.num_breaks,
            'num_failures': self.num_failures
        }
    
    def is_available(self) -> bool:
        """
        Check if the server is available to serve customers.
        
        Returns:
            Whether the server is available
        """
        return self.state == ServerState.IDLE
    
    def is_busy(self) -> bool:
        """
        Check if the server is busy serving a customer.
        
        Returns:
            Whether the server is busy
        """
        return self.state == ServerState.BUSY
    
    def get_state_history(self) -> List[Tuple[float, ServerState]]:
        """
        Get the history of server state transitions.
        
        Returns:
            List of (time, state) tuples, or None if history is not enabled
        """
        return self.state_history
    
    def set_service_rate(self, service_rate: float):
        """
        Change the service rate of the server.
        
        Args:
            service_rate: New service rate (must be positive)
        """
        if service_rate <= 0:
            raise ValueError("Service rate must be positive")
        
        self.service_rate = service_rate

class ServerPool:
    """
    A pool of servers for queueing systems.
    
    This class manages a collection of servers and provides methods for allocating
    servers to customers according to various scheduling policies.
    """
    
    def __init__(self, servers: List[Server]):
        """
        Initialize a server pool.
        
        Args:
            servers: List of servers in the pool
        """
        self.servers = servers
        self.num_servers = len(servers)
    
    def get_available_server(self) -> Optional[Server]:
        """
        Get the first available server in the pool.
        
        Returns:
            An available server, or None if all servers are busy
        """
        for server in self.servers:
            if server.is_available():
                return server
        return None
    
    def get_fastest_available_server(self) -> Optional[Server]:
        """
        Get the fastest available server in the pool.
        
        Returns:
            The fastest available server, or None if all servers are busy
        """
        fastest_server = None
        fastest_rate = 0.0
        
        for server in self.servers:
            if server.is_available() and server.service_rate > fastest_rate:
                fastest_server = server
                fastest_rate = server.service_rate
        
        return fastest_server
    
    def get_server_by_id(self, server_id: int) -> Optional[Server]:
        """
        Get a server by its ID.
        
        Args:
            server_id: ID of the server to retrieve
            
        Returns:
            The server with the given ID, or None if not found
        """
        for server in self.servers:
            if server.id == server_id:
                return server
        return None
    
    def get_number_of_busy_servers(self) -> int:
        """
        Get the number of servers that are currently busy.
        
        Returns:
            Number of busy servers
        """
        return sum(1 for server in self.servers if server.is_busy())
    
    def get_number_of_available_servers(self) -> int:
        """
        Get the number of servers that are currently available.
        
        Returns:
            Number of available servers
        """
        return sum(1 for server in self.servers if server.is_available())
    
    def start_service_on_available_server(self, customer_id: int, current_time: float) -> Optional[Server]:
        """
        Start service for a customer on an available server.
        
        Args:
            customer_id: ID of the customer to serve
            current_time: Current simulation time
            
        Returns:
            The server that started serving the customer, or None if no server is available
        """
        server = self.get_available_server()
        if server is not None:
            if server.start_service(customer_id, current_time):
                return server
        return None
    
    def start_service_on_fastest_server(self, customer_id: int, current_time: float) -> Optional[Server]:
        """
        Start service for a customer on the fastest available server.
        
        Args:
            customer_id: ID of the customer to serve
            current_time: Current simulation time
            
        Returns:
            The server that started serving the customer, or None if no server is available
        """
        server = self.get_fastest_available_server()
        if server is not None:
            if server.start_service(customer_id, current_time):
                return server
        return None
    
    def update_all_servers_time(self, current_time: float):
        """
        Update the time for all servers in the pool.
        
        Args:
            current_time: Current simulation time
        """
        for server in self.servers:
            server.update_time(current_time)
    
    def get_pool_metrics(self) -> Dict[str, float]:
        """
        Get aggregate metrics for the server pool.
        
        Returns:
            Dictionary with metrics such as average utilization, etc.
        """
        if not self.servers:
            return {'num_servers': 0}
        
        # Collect metrics from all servers
        server_metrics = [server.get_metrics() for server in self.servers]
        
        # Calculate aggregated metrics
        metrics = {
            'num_servers': self.num_servers,
            'num_busy_servers': self.get_number_of_busy_servers(),
            'num_available_servers': self.get_number_of_available_servers(),
            'avg_utilization': sum(m['utilization'] for m in server_metrics) / self.num_servers,
            'avg_availability': sum(m['availability'] for m in server_metrics) / self.num_servers,
            'total_services_completed': sum(m['num_services_completed'] for m in server_metrics),
            'total_breaks': sum(m['num_breaks'] for m in server_metrics),
            'total_failures': sum(m['num_failures'] for m in server_metrics),
        }
        
        return metrics 