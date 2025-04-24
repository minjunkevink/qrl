"""
Birth-death process simulation module.
"""

from enum import Enum
from typing import Dict, Optional, Tuple, List, Callable, Any, Union
import numpy as np

from src.simulator.base import CTMCSimulator

class TransitionType(Enum):
    """Enumeration of possible transition types in a birth-death process."""
    BIRTH = "birth"
    DEATH = "death"

class BirthDeathProcess(CTMCSimulator):
    """
    Simulator for a general time-homogeneous birth-death process.
    
    In a birth-death process, the state represents the population size,
    and transitions can only increase or decrease the population by 1.
    """
    
    def __init__(
        self,
        initial_state: int,
        birth_rates: Union[Callable[[int], float], List[float]],
        death_rates: Union[Callable[[int], float], List[float]],
        max_population: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize a birth-death process simulator.
        
        Args:
            initial_state: Initial population size (non-negative integer)
            birth_rates: Either a function mapping population to birth rate,
                         or a list where index i gives birth rate for population i
            death_rates: Either a function mapping population to death rate,
                         or a list where index i gives death rate for population i
            max_population: Maximum allowed population size (unlimited if None)
            seed: Random seed for reproducibility
        """
        if initial_state < 0:
            raise ValueError("Initial state must be non-negative")
        
        super().__init__(initial_state, seed)
        
        self.max_population = max_population
        
        # Configure birth rates
        if callable(birth_rates):
            self.birth_rate_fn = birth_rates
        else:
            birth_rates_array = np.array(birth_rates, dtype=float)
            self.birth_rate_fn = lambda n: birth_rates_array[n] if n < len(birth_rates_array) else 0.0
        
        # Configure death rates
        if callable(death_rates):
            self.death_rate_fn = death_rates
        else:
            death_rates_array = np.array(death_rates, dtype=float)
            self.death_rate_fn = lambda n: death_rates_array[n] if n < len(death_rates_array) else 0.0
    
    def get_transition_rates(self) -> Dict[Tuple[TransitionType, int], float]:
        """
        Get the rates of all possible transitions from the current state.
        
        Returns:
            Dictionary mapping transition tuples (type, new_state) to rates
        """
        current_state = self.state
        transitions = {}
        
        # Birth transition
        if self.max_population is None or current_state < self.max_population:
            birth_rate = self.birth_rate_fn(current_state)
            if birth_rate > 0:
                transitions[(TransitionType.BIRTH, current_state + 1)] = birth_rate
        
        # Death transition
        if current_state > 0:
            death_rate = self.death_rate_fn(current_state)
            if death_rate > 0:
                transitions[(TransitionType.DEATH, current_state - 1)] = death_rate
        
        return transitions
    
    def apply_transition(self, transition: Tuple[TransitionType, int]):
        """
        Apply a transition to the current state.
        
        Args:
            transition: A tuple (transition_type, new_state)
        """
        _, new_state = transition
        self.state = new_state
    
    @staticmethod
    def create_linear_birth_death(
        initial_state: int,
        birth_rate: float,
        death_rate: float,
        max_population: Optional[int] = None,
        seed: Optional[int] = None
    ) -> 'BirthDeathProcess':
        """
        Create a linear birth-death process with constant per-capita rates.
        
        In a linear birth-death process:
        - birth rate = birth_rate * population
        - death rate = death_rate * population
        
        Args:
            initial_state: Initial population size
            birth_rate: Per-capita birth rate
            death_rate: Per-capita death rate
            max_population: Maximum allowed population
            seed: Random seed
            
        Returns:
            A BirthDeathProcess instance with linear rates
        """
        return BirthDeathProcess(
            initial_state=initial_state,
            birth_rates=lambda n: birth_rate * n,
            death_rates=lambda n: death_rate * n,
            max_population=max_population,
            seed=seed
        )
    
    @staticmethod
    def create_immigration_death(
        initial_state: int,
        immigration_rate: float,
        death_rate: float,
        max_population: Optional[int] = None,
        seed: Optional[int] = None
    ) -> 'BirthDeathProcess':
        """
        Create an immigration-death process.
        
        In an immigration-death process:
        - birth rate = immigration_rate (constant, independent of population)
        - death rate = death_rate * population
        
        Args:
            initial_state: Initial population size
            immigration_rate: Constant immigration rate
            death_rate: Per-capita death rate
            max_population: Maximum allowed population
            seed: Random seed
            
        Returns:
            A BirthDeathProcess instance for an immigration-death process
        """
        return BirthDeathProcess(
            initial_state=initial_state,
            birth_rates=lambda n: immigration_rate,
            death_rates=lambda n: death_rate * n,
            max_population=max_population,
            seed=seed
        )
    
    @staticmethod
    def create_logistic_birth_death(
        initial_state: int,
        birth_rate: float,
        death_rate: float,
        carrying_capacity: int,
        seed: Optional[int] = None
    ) -> 'BirthDeathProcess':
        """
        Create a logistic birth-death process with density-dependent birth rates.
        
        In a logistic birth-death process:
        - birth rate = birth_rate * population * (1 - population/carrying_capacity)
        - death rate = death_rate * population
        
        Args:
            initial_state: Initial population size
            birth_rate: Maximum per-capita birth rate
            death_rate: Per-capita death rate
            carrying_capacity: Population carrying capacity
            seed: Random seed
            
        Returns:
            A BirthDeathProcess instance with logistic growth
        """
        def logistic_birth(n):
            return birth_rate * n * (1 - n / carrying_capacity) if n < carrying_capacity else 0.0
        
        return BirthDeathProcess(
            initial_state=initial_state,
            birth_rates=logistic_birth,
            death_rates=lambda n: death_rate * n,
            max_population=carrying_capacity,
            seed=seed
        ) 