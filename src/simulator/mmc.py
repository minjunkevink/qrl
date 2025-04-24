"""
M/M/c queue simulation module.

This module implements the M/M/c queueing system, which is a multi-server queue
with Poisson arrivals and exponential service times.
"""

from enum import Enum
from typing import Dict, Optional, Tuple, List, Any, Union
import numpy as np

from src.simulator.base import CTMCSimulator
from src.simulator.mm1 import QueueEvent

class MMcQueue(CTMCSimulator):
    """
    Simulator for an M/M/c queueing system.
    
    In an M/M/c queue:
    - Arrivals follow a Poisson process with rate λ
    - Service times are exponentially distributed with rate μ per server
    - There are c identical servers operating in parallel
    - The state is the number of customers in the system (including those in service)
    """
    
    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        num_servers: int,
        initial_queue_length: int = 0,
        max_capacity: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize an M/M/c queue simulator.
        
        Args:
            arrival_rate: Customer arrival rate (λ)
            service_rate: Service rate per server (μ)
            num_servers: Number of parallel servers (c)
            initial_queue_length: Initial number of customers in the system
            max_capacity: Maximum queue capacity (unlimited if None)
            seed: Random seed for reproducibility
        """
        if arrival_rate <= 0:
            raise ValueError("Arrival rate must be positive")
        if service_rate <= 0:
            raise ValueError("Service rate must be positive")
        if num_servers <= 0:
            raise ValueError("Number of servers must be positive")
        if initial_queue_length < 0:
            raise ValueError("Initial queue length must be non-negative")
        if max_capacity is not None and initial_queue_length > max_capacity:
            raise ValueError("Initial queue length exceeds maximum capacity")
            
        super().__init__(initial_queue_length, seed)
        
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.num_servers = num_servers
        self.max_capacity = max_capacity
        
        # Metrics
        self.total_arrivals = 0
        self.total_departures = 0
        self.total_service_time = 0.0
        self.total_waiting_time = 0.0
        self.arrival_times = {}  # Dictionary mapping customer ID to arrival time
        self.next_customer_id = 0  # For tracking individual customers
        
        # Server states (0: idle, 1: busy)
        self.server_states = [0] * num_servers
        
        # Begin with arrivals for initial customers
        for _ in range(initial_queue_length):
            self._register_arrival()
            
        # Update server states for initial customers
        self._update_server_states()
    
    def _register_arrival(self):
        """Register a new customer arrival for tracking purposes."""
        customer_id = self.next_customer_id
        self.arrival_times[customer_id] = self.time
        self.next_customer_id += 1
        self.total_arrivals += 1
        return customer_id
    
    def _register_departure(self):
        """Register a customer departure and update statistics."""
        if not self.arrival_times:
            return  # Should not happen, but just in case
        
        # Find the customer who arrived earliest (FIFO)
        customer_id = min(self.arrival_times, key=self.arrival_times.get)
        arrival_time = self.arrival_times.pop(customer_id)
        
        # Calculate waiting and service times
        time_in_system = self.time - arrival_time
        self.total_waiting_time += time_in_system
        self.total_service_time += min(time_in_system, self.time)  # Approximate service time
        self.total_departures += 1
    
    def _update_server_states(self):
        """Update the state of each server based on the current queue length."""
        queue_length = self.state
        # Reset all servers to idle
        self.server_states = [0] * self.num_servers
        
        # Assign customers to servers (up to the number of servers)
        for i in range(min(queue_length, self.num_servers)):
            self.server_states[i] = 1
    
    def get_transition_rates(self) -> Dict[Tuple[QueueEvent, int], float]:
        """
        Get the rates of all possible transitions from the current state.
        
        Returns:
            Dictionary mapping transition tuples (event, new_state) to rates
        """
        queue_length = self.state
        transitions = {}
        
        # Arrival transition
        if self.max_capacity is None or queue_length < self.max_capacity:
            transitions[(QueueEvent.ARRIVAL, queue_length + 1)] = self.arrival_rate
        
        # Departure transition
        if queue_length > 0:
            # Total service rate depends on the number of busy servers
            num_busy_servers = min(queue_length, self.num_servers)
            total_service_rate = num_busy_servers * self.service_rate
            transitions[(QueueEvent.DEPARTURE, queue_length - 1)] = total_service_rate
        
        return transitions
    
    def apply_transition(self, transition: Tuple[QueueEvent, int]):
        """
        Apply a transition to the current state.
        
        Args:
            transition: A tuple (event, new_state)
        """
        event, new_state = transition
        
        if event == QueueEvent.ARRIVAL:
            self._register_arrival()
        elif event == QueueEvent.DEPARTURE:
            self._register_departure()
        elif event == QueueEvent.SERVICE_RATE_CHANGE:
            # This event doesn't change the state, just the service rate
            pass
        
        self.state = new_state
        self._update_server_states()
    
    def set_service_rate(self, new_rate: float):
        """
        Change the service rate of all servers.
        
        This method can be used for dynamic control of the queue.
        
        Args:
            new_rate: New service rate per server (must be positive)
        """
        if new_rate <= 0:
            raise ValueError("Service rate must be positive")
        
        self.service_rate = new_rate
    
    def set_arrival_rate(self, new_rate: float):
        """
        Change the arrival rate of the queue.
        
        This method can be used to simulate changing arrival patterns.
        
        Args:
            new_rate: New arrival rate (must be positive)
        """
        if new_rate <= 0:
            raise ValueError("Arrival rate must be positive")
        
        self.arrival_rate = new_rate
    
    def set_num_servers(self, num_servers: int):
        """
        Change the number of servers in the system.
        
        This method can be used for dynamic resource allocation.
        
        Args:
            num_servers: New number of servers (must be positive)
        """
        if num_servers <= 0:
            raise ValueError("Number of servers must be positive")
        
        self.num_servers = num_servers
        self._update_server_states()
    
    def get_queue_length(self) -> int:
        """
        Get the current queue length.
        
        Returns:
            Number of customers in the system
        """
        return self.state
    
    def get_waiting_queue_length(self) -> int:
        """
        Get the number of customers waiting in the queue (not being served).
        
        Returns:
            Number of customers waiting for service
        """
        return max(0, self.state - self.num_servers)
    
    def get_utilization(self) -> float:
        """
        Calculate the current utilization of the system.
        
        Returns:
            The ratio λ/(c*μ) or 1.0 if the system is overloaded
        """
        utilization = self.arrival_rate / (self.num_servers * self.service_rate)
        return min(utilization, 1.0)
    
    def get_server_utilization(self) -> float:
        """
        Calculate the fraction of busy servers.
        
        Returns:
            The fraction of servers that are currently busy
        """
        if self.num_servers == 0:
            return 0.0
        return sum(self.server_states) / self.num_servers
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics of the queue.
        
        Returns:
            Dictionary with metrics such as average waiting time,
            throughput, utilization, etc.
        """
        metrics = {
            'queue_length': self.state,
            'waiting_queue_length': self.get_waiting_queue_length(),
            'system_utilization': self.get_utilization(),
            'server_utilization': self.get_server_utilization(),
            'total_arrivals': self.total_arrivals,
            'total_departures': self.total_departures,
            'num_busy_servers': sum(self.server_states),
            'num_servers': self.num_servers
        }
        
        # Add time-based metrics only if some events have occurred
        if self.total_departures > 0:
            metrics['average_waiting_time'] = self.total_waiting_time / self.total_departures
            metrics['average_service_time'] = self.total_service_time / self.total_departures
            metrics['throughput'] = self.total_departures / self.time if self.time > 0 else 0
        
        return metrics
    
    def reset(self, initial_queue_length: Optional[int] = None, seed: Optional[int] = None):
        """
        Reset the queue to an initial state.
        
        Args:
            initial_queue_length: New initial queue length (uses the current parameters if None)
            seed: New random seed (uses the current one if None)
        """
        if initial_queue_length is not None:
            if initial_queue_length < 0:
                raise ValueError("Initial queue length must be non-negative")
            if self.max_capacity is not None and initial_queue_length > self.max_capacity:
                raise ValueError("Initial queue length exceeds maximum capacity")
        
        # Use parent class reset for common variables
        super().reset(initial_queue_length if initial_queue_length is not None else self.state, seed)
        
        # Reset queue-specific metrics
        self.total_arrivals = 0
        self.total_departures = 0
        self.total_service_time = 0.0
        self.total_waiting_time = 0.0
        self.arrival_times = {}
        self.next_customer_id = 0
        
        # Reset server states
        self.server_states = [0] * self.num_servers
        
        # Begin with arrivals for initial customers
        for _ in range(self.state):
            self._register_arrival()
            
        # Update server states for initial customers
        self._update_server_states() 