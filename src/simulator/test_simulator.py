#!/usr/bin/env python3
"""
Test script for the birth-death process simulator.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.simulator.base import CTMCSimulator
from src.simulator.birth_death import BirthDeathProcess, TransitionType
from src.utils.visualization import (
    plot_trajectory, 
    plot_multiple_trajectories, 
    animate_trajectory,
    plot_distribution
)

def test_linear_birth_death():
    """Test a linear birth-death process."""
    print("\n==== Testing Linear Birth-Death Process ====")
    
    # Parameters
    initial_state = 10
    birth_rate = 0.5   # per capita birth rate
    death_rate = 0.3   # per capita death rate
    max_time = 20.0
    seed = 42
    
    # Create the process
    process = BirthDeathProcess.create_linear_birth_death(
        initial_state=initial_state,
        birth_rate=birth_rate,
        death_rate=death_rate,
        seed=seed
    )
    
    # Run the simulation
    print(f"Running linear birth-death process with initial state {initial_state}")
    print(f"Per-capita birth rate: {birth_rate}, per-capita death rate: {death_rate}")
    
    start_time = time.time()
    history = process.run(max_time=max_time)
    run_time = time.time() - start_time
    
    print(f"Simulation completed in {run_time:.4f} seconds")
    print(f"Final state: {process.get_state()}")
    print(f"Number of transitions: {len(history) - 1}")
    
    # Plot the trajectory
    plot_trajectory(
        history, 
        title=f"Linear Birth-Death Process (λ={birth_rate}, μ={death_rate})",
        xlabel="Time",
        ylabel="Population",
        figsize=(10, 6)
    )
    
    return process, history

def test_immigration_death():
    """Test an immigration-death process."""
    print("\n==== Testing Immigration-Death Process ====")
    
    # Parameters
    initial_state = 0
    immigration_rate = 2.0   # constant immigration rate
    death_rate = 0.25        # per capita death rate
    max_time = 25.0
    seed = 42
    
    # Create the process
    process = BirthDeathProcess.create_immigration_death(
        initial_state=initial_state,
        immigration_rate=immigration_rate,
        death_rate=death_rate,
        seed=seed
    )
    
    # Run the simulation
    print(f"Running immigration-death process with initial state {initial_state}")
    print(f"Immigration rate: {immigration_rate}, per-capita death rate: {death_rate}")
    
    start_time = time.time()
    history = process.run(max_time=max_time)
    run_time = time.time() - start_time
    
    print(f"Simulation completed in {run_time:.4f} seconds")
    print(f"Final state: {process.get_state()}")
    print(f"Number of transitions: {len(history) - 1}")
    
    # Plot the trajectory
    plot_trajectory(
        history, 
        title=f"Immigration-Death Process (λ={immigration_rate}, μ={death_rate})",
        xlabel="Time",
        ylabel="Population",
        figsize=(10, 6)
    )
    
    return process, history

def test_logistic_birth_death():
    """Test a logistic birth-death process."""
    print("\n==== Testing Logistic Birth-Death Process ====")
    
    # Parameters
    initial_state = 5
    birth_rate = 0.8        # maximum per capita birth rate
    death_rate = 0.2        # per capita death rate
    carrying_capacity = 50
    max_time = 30.0
    seed = 42
    
    # Create the process
    process = BirthDeathProcess.create_logistic_birth_death(
        initial_state=initial_state,
        birth_rate=birth_rate,
        death_rate=death_rate,
        carrying_capacity=carrying_capacity,
        seed=seed
    )
    
    # Run the simulation
    print(f"Running logistic birth-death process with initial state {initial_state}")
    print(f"Max birth rate: {birth_rate}, death rate: {death_rate}, carrying capacity: {carrying_capacity}")
    
    start_time = time.time()
    history = process.run(max_time=max_time)
    run_time = time.time() - start_time
    
    print(f"Simulation completed in {run_time:.4f} seconds")
    print(f"Final state: {process.get_state()}")
    print(f"Number of transitions: {len(history) - 1}")
    
    # Plot the trajectory
    plot_trajectory(
        history, 
        title=f"Logistic Birth-Death Process (r={birth_rate}, μ={death_rate}, K={carrying_capacity})",
        xlabel="Time",
        ylabel="Population",
        figsize=(10, 6)
    )
    
    return process, history

def test_multiple_trajectories():
    """Test simulating and plotting multiple trajectories."""
    print("\n==== Testing Multiple Trajectories ====")
    
    # Parameters
    num_trajectories = 10
    initial_state = 10
    birth_rate = 0.5
    death_rate = 0.3
    max_time = 20.0
    
    histories = []
    final_states = []
    
    # Run multiple simulations
    print(f"Running {num_trajectories} independent trajectories...")
    
    start_time = time.time()
    for i in range(num_trajectories):
        # Create a process with a different seed for each trajectory
        process = BirthDeathProcess.create_linear_birth_death(
            initial_state=initial_state,
            birth_rate=birth_rate,
            death_rate=death_rate,
            seed=i
        )
        
        # Run the simulation
        history = process.run(max_time=max_time)
        histories.append(history)
        final_states.append(process.get_state())
    
    run_time = time.time() - start_time
    print(f"All simulations completed in {run_time:.4f} seconds")
    
    # Plot all trajectories
    plot_multiple_trajectories(
        histories,
        title=f"Multiple Linear Birth-Death Processes (λ={birth_rate}, μ={death_rate})",
        xlabel="Time",
        ylabel="Population",
        figsize=(12, 8)
    )
    
    # Plot the distribution of final states
    plot_distribution(
        final_states,
        title=f"Distribution of Final States at t={max_time}",
        xlabel="Population",
        ylabel="Frequency",
        bins=range(min(final_states), max(final_states) + 2),
        figsize=(10, 6)
    )
    
    return histories, final_states

def test_animation():
    """Test creating an animation of a birth-death process."""
    print("\n==== Testing Trajectory Animation ====")
    
    # Parameters
    initial_state = 5
    immigration_rate = 3.0
    death_rate = 0.2
    max_time = 30.0
    seed = 123
    
    # Create and run the process
    process = BirthDeathProcess.create_immigration_death(
        initial_state=initial_state,
        immigration_rate=immigration_rate,
        death_rate=death_rate,
        seed=seed
    )
    
    history = process.run(max_time=max_time)
    print(f"Generated trajectory with {len(history)} points")
    
    # Create animation
    print("Creating animation...")
    anim = animate_trajectory(
        history,
        title=f"Immigration-Death Process (λ={immigration_rate}, μ={death_rate})",
        xlabel="Time",
        ylabel="Population",
        interval=100,  # ms between frames
        trail_length=20
    )
    
    print("Animation created. Close the plot window to continue.")
    plt.show()
    
    return anim

def run_tests():
    """Run all tests."""
    # Create output directory for plots
    output_dir = Path("output/test_simulator")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test linear birth-death process
    process1, history1 = test_linear_birth_death()
    
    # Test immigration-death process
    process2, history2 = test_immigration_death()
    
    # Test logistic birth-death process
    process3, history3 = test_logistic_birth_death()
    
    # Test multiple trajectories
    histories, final_states = test_multiple_trajectories()
    
    # Test animation (only if in interactive mode)
    if plt.isinteractive():
        anim = test_animation()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    # Create a directory for output
    os.makedirs("output", exist_ok=True)
    
    # Run the tests
    run_tests() 