#!/usr/bin/env python3
"""
Test script for the queueing system simulators.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.simulator.mm1 import MM1Queue, QueueEvent
from src.simulator.mmc import MMcQueue
from src.simulator.server import Server, ServerState, ServerPool
from src.utils.visualization import (
    plot_trajectory, 
    plot_multiple_trajectories, 
    plot_heatmap
)

def mm1_basic_test():
    """Test a basic M/M/1 queue with analytical validation."""
    print("\n==== Testing M/M/1 Queue ====")
    
    # Parameters
    arrival_rate = 2.0  # Customers per unit time
    service_rate = 3.0  # Customers per unit time
    initial_queue_length = 0
    max_time = 100.0
    seed = 42
    
    # Create the queue
    queue = MM1Queue(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        initial_queue_length=initial_queue_length,
        seed=seed
    )
    
    # Run the simulation
    print(f"Running M/M/1 queue with λ={arrival_rate}, μ={service_rate}")
    start_time = time.time()
    history = queue.run(max_time=max_time)
    run_time = time.time() - start_time
    
    # Get simulation results
    metrics = queue.get_metrics()
    
    print(f"Simulation completed in {run_time:.4f} seconds")
    print(f"Final queue length: {metrics['queue_length']}")
    print(f"Utilization: {metrics['utilization']:.4f}")
    print(f"Total arrivals: {metrics['total_arrivals']}")
    print(f"Total departures: {metrics['total_departures']}")
    
    if 'average_waiting_time' in metrics:
        print(f"Average time in system: {metrics['average_waiting_time']:.4f}")
        
        # Compare with M/M/1 analytical results
        theoretical_utilization = arrival_rate / service_rate
        theoretical_avg_time = 1 / (service_rate - arrival_rate)
        
        print(f"Theoretical utilization: {theoretical_utilization:.4f}")
        print(f"Theoretical average time in system: {theoretical_avg_time:.4f}")
        
        # Calculate relative error
        utilization_error = abs(metrics['utilization'] - theoretical_utilization) / theoretical_utilization
        time_error = abs(metrics['average_waiting_time'] - theoretical_avg_time) / theoretical_avg_time
        
        print(f"Utilization relative error: {utilization_error:.2%}")
        print(f"Average time relative error: {time_error:.2%}")
    
    # Plot queue length trajectory
    plot_trajectory(
        history, 
        title=f"M/M/1 Queue Length (λ={arrival_rate}, μ={service_rate})",
        xlabel="Time",
        ylabel="Queue Length",
        figsize=(10, 6)
    )
    
    return queue, history

def mm1_control_test():
    """Test an M/M/1 queue with dynamic service rate control."""
    print("\n==== Testing M/M/1 Queue with Control ====")
    
    # Parameters
    arrival_rate = 2.0  # Customers per unit time
    initial_service_rate = 3.0  # Customers per unit time
    max_time = 100.0
    seed = 42
    
    # Create the queue
    queue = MM1Queue(
        arrival_rate=arrival_rate,
        service_rate=initial_service_rate,
        initial_queue_length=0,
        seed=seed
    )
    
    # Control policy: Increase service rate when queue length exceeds threshold
    def control_policy(simulator, dt, transition):
        """Dynamically adjust service rate based on queue length."""
        queue_length = simulator.state
        
        # If queue length is high, increase service rate
        if queue_length > 10:
            simulator.set_service_rate(5.0)
        # If queue length is moderate, use medium service rate
        elif queue_length > 5:
            simulator.set_service_rate(4.0)
        # If queue length is low, use base service rate
        else:
            simulator.set_service_rate(3.0)
    
    # Run the simulation with control
    print(f"Running M/M/1 queue with λ={arrival_rate}, initial μ={initial_service_rate}, and dynamic control")
    start_time = time.time()
    history = queue.run(max_time=max_time, callback=control_policy)
    run_time = time.time() - start_time
    
    # Get simulation results
    metrics = queue.get_metrics()
    
    print(f"Simulation completed in {run_time:.4f} seconds")
    print(f"Final queue length: {metrics['queue_length']}")
    print(f"Utilization: {metrics['utilization']:.4f}")
    print(f"Total arrivals: {metrics['total_arrivals']}")
    print(f"Total departures: {metrics['total_departures']}")
    
    if 'average_waiting_time' in metrics:
        print(f"Average time in system: {metrics['average_waiting_time']:.4f}")
    
    # Plot queue length trajectory
    plot_trajectory(
        history, 
        title=f"M/M/1 Queue with Dynamic Control (λ={arrival_rate})",
        xlabel="Time",
        ylabel="Queue Length",
        figsize=(10, 6)
    )
    
    return queue, history

def mmc_basic_test():
    """Test an M/M/c queue with multiple servers."""
    print("\n==== Testing M/M/c Queue ====")
    
    # Parameters
    arrival_rate = 5.0  # Customers per unit time
    service_rate = 1.5  # Customers per unit time per server
    num_servers = 4
    initial_queue_length = 0
    max_time = 100.0
    seed = 42
    
    # Create the queue
    queue = MMcQueue(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        num_servers=num_servers,
        initial_queue_length=initial_queue_length,
        seed=seed
    )
    
    # Run the simulation
    print(f"Running M/M/{num_servers} queue with λ={arrival_rate}, μ={service_rate} per server")
    start_time = time.time()
    history = queue.run(max_time=max_time)
    run_time = time.time() - start_time
    
    # Get simulation results
    metrics = queue.get_metrics()
    
    print(f"Simulation completed in {run_time:.4f} seconds")
    print(f"Final queue length: {metrics['queue_length']}")
    print(f"Final waiting queue length: {metrics['waiting_queue_length']}")
    print(f"System utilization: {metrics['system_utilization']:.4f}")
    print(f"Server utilization: {metrics['server_utilization']:.4f}")
    print(f"Number of busy servers: {metrics['num_busy_servers']} / {metrics['num_servers']}")
    print(f"Total arrivals: {metrics['total_arrivals']}")
    print(f"Total departures: {metrics['total_departures']}")
    
    if 'average_waiting_time' in metrics:
        print(f"Average time in system: {metrics['average_waiting_time']:.4f}")
        
        # Compare with M/M/c analytical results (approximation)
        rho = arrival_rate / (num_servers * service_rate)
        theoretical_utilization = rho
        
        # For stable system (rho < 1)
        if rho < 1:
            # This is a simplification; exact formula is more complex
            theoretical_avg_time = 1 / service_rate * (1 + rho / (1 - rho))
            print(f"Theoretical utilization: {theoretical_utilization:.4f}")
            print(f"Approximate theoretical average time in system: {theoretical_avg_time:.4f}")
    
    # Plot queue length trajectory
    plot_trajectory(
        history, 
        title=f"M/M/{num_servers} Queue Length (λ={arrival_rate}, μ={service_rate} per server)",
        xlabel="Time",
        ylabel="Queue Length",
        figsize=(10, 6)
    )
    
    return queue, history

def mmc_dynamic_servers_test():
    """Test an M/M/c queue with a dynamic number of servers."""
    print("\n==== Testing M/M/c Queue with Dynamic Number of Servers ====")
    
    # Parameters
    arrival_rate = 7.0  # Customers per unit time
    service_rate = 1.5  # Customers per unit time per server
    initial_servers = 2
    max_servers = 6
    initial_queue_length = 0
    max_time = 100.0
    seed = 42
    
    # Create the queue
    queue = MMcQueue(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        num_servers=initial_servers,
        initial_queue_length=initial_queue_length,
        seed=seed
    )
    
    # Control policy: Adjust number of servers based on queue length
    def control_policy(simulator, dt, transition):
        """Dynamically adjust number of servers based on queue length."""
        queue_length = simulator.state
        current_servers = simulator.num_servers
        
        # Add servers if queue is building up
        if queue_length > 15 and current_servers < max_servers:
            simulator.set_num_servers(current_servers + 1)
            print(f"Time {simulator.time:.2f}: Increasing servers to {current_servers + 1}")
        # Remove servers if queue is small and we have more than minimum servers
        elif queue_length < 3 and current_servers > 2:
            simulator.set_num_servers(current_servers - 1)
            print(f"Time {simulator.time:.2f}: Decreasing servers to {current_servers - 1}")
    
    # Run the simulation with control
    print(f"Running M/M/c queue with λ={arrival_rate}, μ={service_rate} per server, initial c={initial_servers}")
    start_time = time.time()
    history = queue.run(max_time=max_time, callback=control_policy)
    run_time = time.time() - start_time
    
    # Get simulation results
    metrics = queue.get_metrics()
    
    print(f"Simulation completed in {run_time:.4f} seconds")
    print(f"Final queue length: {metrics['queue_length']}")
    print(f"Final number of servers: {metrics['num_servers']}")
    print(f"System utilization: {metrics['system_utilization']:.4f}")
    print(f"Server utilization: {metrics['server_utilization']:.4f}")
    print(f"Number of busy servers: {metrics['num_busy_servers']} / {metrics['num_servers']}")
    print(f"Total arrivals: {metrics['total_arrivals']}")
    print(f"Total departures: {metrics['total_departures']}")
    
    if 'average_waiting_time' in metrics:
        print(f"Average time in system: {metrics['average_waiting_time']:.4f}")
    
    # Plot queue length trajectory
    plot_trajectory(
        history, 
        title=f"M/M/c Queue with Dynamic Servers (λ={arrival_rate}, μ={service_rate} per server)",
        xlabel="Time",
        ylabel="Queue Length",
        figsize=(10, 6)
    )
    
    return queue, history

def test_server_with_breaks():
    """Test the Server class with breaks and state transitions."""
    print("\n==== Testing Server with Breaks ====")
    
    # Create a server
    server = Server(
        id=1,
        service_rate=2.0,
        break_rate=0.1,
        break_duration=5.0,
        state_history=True
    )
    
    # Simulate a sequence of events
    print("Simulating server state transitions...")
    
    # Start with server idle
    print(f"Time 0.0: Server state is {server.state.value}")
    
    # Customer 1 arrives
    server.start_service(customer_id=1, current_time=1.0)
    print(f"Time 1.0: Started service for customer 1, state is {server.state.value}")
    
    # Service completion
    customer_id = server.complete_service(current_time=1.5)
    print(f"Time 1.5: Completed service for customer {customer_id}, state is {server.state.value}")
    
    # Server takes a break
    server.start_break(current_time=2.0)
    print(f"Time 2.0: Started break, state is {server.state.value}")
    
    # Break ends
    server.end_break(current_time=7.0)
    print(f"Time 7.0: Ended break, state is {server.state.value}")
    
    # Customer 2 arrives
    server.start_service(customer_id=2, current_time=8.0)
    print(f"Time 8.0: Started service for customer 2, state is {server.state.value}")
    
    # Server fails
    server.fail(current_time=8.5)
    print(f"Time 8.5: Server failed, state is {server.state.value}")
    
    # Start repair
    server.start_repair(current_time=9.0)
    print(f"Time 9.0: Started repair, state is {server.state.value}")
    
    # Complete repair
    server.complete_repair(current_time=12.0)
    print(f"Time 12.0: Completed repair, state is {server.state.value}")
    
    # Update time to end
    server.update_time(current_time=15.0)
    
    # Get metrics
    metrics = server.get_metrics()
    print("\nServer Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Get state history
    history = server.get_state_history()
    times = [t for t, _ in history]
    states = [s.value for _, s in history]
    
    # Plot state history
    plt.figure(figsize=(10, 6))
    plt.step(times, states, where='post', marker='o')
    plt.yticks([s.value for s in ServerState])
    plt.title("Server State History")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return server, history

def test_server_pool():
    """Test the ServerPool class with heterogeneous servers."""
    print("\n==== Testing Server Pool ====")
    
    # Create several servers with different characteristics
    servers = [
        Server(id=1, service_rate=1.5, break_rate=0.05, break_duration=2.0, state_history=True),
        Server(id=2, service_rate=2.0, failure_rate=0.02, repair_time=3.0, state_history=True),
        Server(id=3, service_rate=1.8, setup_time=0.5, state_history=True),
        Server(id=4, service_rate=2.2, state_history=True)
    ]
    
    # Create a server pool
    pool = ServerPool(servers)
    
    # Simulate some customer arrivals and service completions
    print("Simulating customer arrivals and service completions...")
    
    # Start with all servers idle
    print(f"Time 0.0: Available servers: {pool.get_number_of_available_servers()}/{pool.num_servers}")
    
    # Customer 1 arrives - assign to fastest server
    server1 = pool.start_service_on_fastest_server(customer_id=1, current_time=1.0)
    print(f"Time 1.0: Customer 1 assigned to server {server1.id} (rate={server1.service_rate})")
    
    # Customer 2 arrives - assign to fastest available server
    server2 = pool.start_service_on_fastest_server(customer_id=2, current_time=1.5)
    print(f"Time 1.5: Customer 2 assigned to server {server2.id} (rate={server2.service_rate})")
    
    # Customer 3 arrives - assign to fastest available server
    server3 = pool.start_service_on_fastest_server(customer_id=3, current_time=2.0)
    print(f"Time 2.0: Customer 3 assigned to server {server3.id} (rate={server3.service_rate})")
    
    # Customer 4 arrives - assign to fastest available server
    server4 = pool.start_service_on_fastest_server(customer_id=4, current_time=2.5)
    print(f"Time 2.5: Customer 4 assigned to server {server4.id} (rate={server4.service_rate})")
    
    # Customer 5 arrives - no server available
    server5 = pool.start_service_on_fastest_server(customer_id=5, current_time=3.0)
    print(f"Time 3.0: Customer 5 assigned to server {server5 if server5 else 'None - all busy'}")
    
    # Server 1 completes service
    customer_id = server1.complete_service(current_time=4.0)
    print(f"Time 4.0: Server {server1.id} completed service for customer {customer_id}")
    
    # Customer 6 arrives - assign to newly available server
    server6 = pool.start_service_on_available_server(customer_id=6, current_time=4.5)
    print(f"Time 4.5: Customer 6 assigned to server {server6.id if server6 else 'None'}")
    
    # Server 2 fails
    server2.fail(current_time=5.0)
    print(f"Time 5.0: Server {server2.id} failed while serving a customer")
    
    # Server 2 starts repair
    server2.start_repair(current_time=5.1)
    print(f"Time 5.1: Started repair for server {server2.id}")
    
    # Server 3 completes service and goes to setup
    customer_id = server3.complete_service(current_time=6.0)
    print(f"Time 6.0: Server {server3.id} completed service for customer {customer_id} and is in setup")
    
    # Server 3 completes setup
    server3.complete_setup(current_time=6.5)
    print(f"Time 6.5: Server {server3.id} completed setup and is idle")
    
    # Server 4 completes service
    customer_id = server4.complete_service(current_time=7.0)
    print(f"Time 7.0: Server {server4.id} completed service for customer {customer_id}")
    
    # Server 2 completes repair
    server2.complete_repair(current_time=8.0)
    print(f"Time 8.0: Server {server2.id} completed repair and is idle")
    
    # Update all servers to final time
    pool.update_all_servers_time(current_time=10.0)
    
    # Get pool metrics
    metrics = pool.get_pool_metrics()
    print("\nServer Pool Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Plot state histories for all servers
    plt.figure(figsize=(12, 8))
    for i, server in enumerate(servers):
        history = server.get_state_history()
        times = [t for t, _ in history]
        states = [s.value for _, s in history]
        plt.subplot(len(servers), 1, i+1)
        plt.step(times, states, where='post', marker='o')
        plt.yticks([s.value for s in ServerState])
        plt.title(f"Server {server.id} (rate={server.service_rate})")
        if i == len(servers) - 1:
            plt.xlabel("Time")
        plt.ylabel("State")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pool, servers

def run_tests():
    """Run all tests."""
    # Create output directory for plots
    output_dir = Path("output/test_queues")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test M/M/1 queue
    mm1_queue, mm1_history = mm1_basic_test()
    
    # Test M/M/1 queue with control
    mm1_control_queue, mm1_control_history = mm1_control_test()
    
    # Test M/M/c queue
    mmc_queue, mmc_history = mmc_basic_test()
    
    # Test M/M/c queue with dynamic servers
    mmc_dynamic_queue, mmc_dynamic_history = mmc_dynamic_servers_test()
    
    # Test Server with breaks
    server, server_history = test_server_with_breaks()
    
    # Test ServerPool
    pool, servers = test_server_pool()
    
    print("\nAll tests completed successfully!")
    
if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run the tests
    run_tests() 