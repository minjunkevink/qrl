"""
Base class for continuous-time Markov chain (CTMC) simulators.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import time
from abc import ABC, abstractmethod

class CTMCSimulator(ABC):
    """
    Base class for continuous-time Markov chain simulators.
    
    This abstract class provides the framework for simulating
    continuous-time Markov processes using the Gillespie algorithm.
    """
    
    def __init__(self, initial_state: Any, seed: Optional[int] = None):
        """
        Initialize the CTMC simulator.
        
        Args:
            initial_state: The initial state of the system
            seed: Random seed for reproducibility
        """
        self.state = initial_state
        self.time = 0.0
        self.history = []
        self.rng = np.random.RandomState(seed)
        self.save_history = False
        self._record_state()
    
    def _record_state(self):
        """Record the current state and time to history."""
        if self.save_history:
            self.history.append((self.time, self.state))
    
    @abstractmethod
    def get_transition_rates(self) -> Dict[Any, float]:
        """
        Return a dictionary mapping possible state transitions to their rates.
        
        Returns:
            A dictionary where keys are possible transitions and values are rates
        """
        pass
    
    @abstractmethod
    def apply_transition(self, transition: Any):
        """
        Apply a transition to the current state.
        
        Args:
            transition: The transition to apply
        """
        pass
    
    def step(self) -> Tuple[float, Optional[Any]]:
        """
        Execute a single step of the simulation using Gillespie algorithm.
        
        Returns:
            Tuple of (time increment, transition applied or None if no transition occurred)
        """
        # Get possible transitions and their rates
        transitions = self.get_transition_rates()
        
        # If no transitions are possible, return
        if not transitions or sum(transitions.values()) == 0:
            return 0.0, None
        
        # Calculate total rate
        total_rate = sum(transitions.values())
        
        # Sample time increment from exponential distribution
        dt = self.rng.exponential(scale=1.0/total_rate)
        self.time += dt
        
        # Sample transition with probability proportional to its rate
        transitions_list = list(transitions.items())
        transitions_array = np.array([rate for _, rate in transitions_list])
        transitions_prob = transitions_array / total_rate
        
        idx = self.rng.choice(len(transitions_list), p=transitions_prob)
        selected_transition, _ = transitions_list[idx]
        
        # Apply the transition
        self.apply_transition(selected_transition)
        self._record_state()
        
        return dt, selected_transition
    
    def run(self, max_time: float = None, max_steps: int = None, 
            callback: Optional[Callable] = None) -> List[Tuple[float, Any]]:
        """
        Run the simulation for a specified amount of time or number of steps.
        
        Args:
            max_time: Maximum simulation time
            max_steps: Maximum number of simulation steps
            callback: Optional callback function called after each step
                      with signature callback(simulator, dt, transition)
        
        Returns:
            List of (time, state) tuples if save_history is True, otherwise empty list
        """
        if max_time is None and max_steps is None:
            raise ValueError("Either max_time or max_steps must be specified")
        
        step_count = 0
        
        self.history = []
        self.save_history = True
        self._record_state()  # Record initial state
        
        while True:
            # Check termination conditions
            if max_time is not None and self.time >= max_time:
                break
            if max_steps is not None and step_count >= max_steps:
                break
            
            # Execute a step
            dt, transition = self.step()
            step_count += 1
            
            # Call callback if provided
            if callback:
                callback(self, dt, transition)
            
            # If no transition was possible, break
            if transition is None:
                break
        
        self.save_history = False
        return self.history
    
    def reset(self, initial_state: Any = None, seed: Optional[int] = None):
        """
        Reset the simulator to an initial state.
        
        Args:
            initial_state: New initial state (uses the current one if None)
            seed: New random seed (uses the current one if None)
        """
        if initial_state is not None:
            self.state = initial_state
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            
        self.time = 0.0
        self.history = []
        self._record_state()
        
    def get_state(self) -> Any:
        """
        Get the current state of the system.
        
        Returns:
            The current state
        """
        return self.state
    
    def get_time(self) -> float:
        """
        Get the current simulation time.
        
        Returns:
            The current simulation time
        """
        return self.time 