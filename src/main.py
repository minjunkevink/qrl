#!/usr/bin/env python3
"""
Main entry point for the Birth-Death RL project.

This script provides a command-line interface to run different components
of the project, including simulations, training RL agents, and evaluations.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.simulator.mm1 import MM1Queue
from src.simulator.mmc import MMcQueue
from src.simulator.server import Server, ServerPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_mm1_simulation(args):
    """Run an M/M/1 queue simulation."""
    logger.info(f"Running M/M/1 queue simulation with λ={args.arrival_rate}, μ={args.service_rate}")
    
    # Create the queue
    queue = MM1Queue(
        arrival_rate=args.arrival_rate,
        service_rate=args.service_rate,
        initial_queue_length=args.initial_queue_length,
        max_capacity=args.max_capacity,
        seed=args.seed
    )
    
    # Run the simulation
    start_time = time.time()
    history = queue.run(max_time=args.max_time)
    run_time = time.time() - start_time
    
    # Get and display metrics
    metrics = queue.get_metrics()
    logger.info(f"Simulation completed in {run_time:.4f} seconds")
    logger.info(f"Final queue length: {metrics['queue_length']}")
    logger.info(f"Utilization: {metrics['utilization']:.4f}")
    logger.info(f"Total arrivals: {metrics['total_arrivals']}")
    logger.info(f"Total departures: {metrics['total_departures']}")
    
    if 'average_waiting_time' in metrics:
        logger.info(f"Average time in system: {metrics['average_waiting_time']:.4f}")
        
        # Compare with M/M/1 analytical results
        theoretical_utilization = args.arrival_rate / args.service_rate
        theoretical_avg_time = 1 / (args.service_rate - args.arrival_rate)
        
        logger.info(f"Theoretical utilization: {theoretical_utilization:.4f}")
        logger.info(f"Theoretical average time in system: {theoretical_avg_time:.4f}")

def run_mmc_simulation(args):
    """Run an M/M/c queue simulation."""
    logger.info(f"Running M/M/{args.num_servers} queue simulation with λ={args.arrival_rate}, μ={args.service_rate}")
    
    # Create the queue
    queue = MMcQueue(
        arrival_rate=args.arrival_rate,
        service_rate=args.service_rate,
        num_servers=args.num_servers,
        initial_queue_length=args.initial_queue_length,
        max_capacity=args.max_capacity,
        seed=args.seed
    )
    
    # Run the simulation
    start_time = time.time()
    history = queue.run(max_time=args.max_time)
    run_time = time.time() - start_time
    
    # Get and display metrics
    metrics = queue.get_metrics()
    logger.info(f"Simulation completed in {run_time:.4f} seconds")
    logger.info(f"Final queue length: {metrics['queue_length']}")
    logger.info(f"Final waiting queue length: {metrics['waiting_queue_length']}")
    logger.info(f"System utilization: {metrics['system_utilization']:.4f}")
    logger.info(f"Server utilization: {metrics['server_utilization']:.4f}")
    logger.info(f"Number of busy servers: {metrics['num_busy_servers']} / {metrics['num_servers']}")
    logger.info(f"Total arrivals: {metrics['total_arrivals']}")
    logger.info(f"Total departures: {metrics['total_departures']}")
    
    if 'average_waiting_time' in metrics:
        logger.info(f"Average time in system: {metrics['average_waiting_time']:.4f}")

def run_tests(args):
    """Run the test suite for queue simulators."""
    from src.simulator.test_queues import run_tests
    logger.info("Running test suite for queue simulators")
    run_tests()

def main():
    """Main entry point for the program."""
    parser = argparse.ArgumentParser(description="Birth-Death RL: A reinforcement learning framework for birth-death processes")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # MM1 simulation parser
    mm1_parser = subparsers.add_parser("mm1", help="Run M/M/1 queue simulation")
    mm1_parser.add_argument("--arrival-rate", type=float, default=2.0, help="Arrival rate (lambda)")
    mm1_parser.add_argument("--service-rate", type=float, default=3.0, help="Service rate (mu)")
    mm1_parser.add_argument("--initial-queue-length", type=int, default=0, help="Initial queue length")
    mm1_parser.add_argument("--max-capacity", type=int, default=None, help="Maximum queue capacity")
    mm1_parser.add_argument("--max-time", type=float, default=100.0, help="Maximum simulation time")
    mm1_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    mm1_parser.set_defaults(func=run_mm1_simulation)
    
    # MMc simulation parser
    mmc_parser = subparsers.add_parser("mmc", help="Run M/M/c queue simulation")
    mmc_parser.add_argument("--arrival-rate", type=float, default=5.0, help="Arrival rate (lambda)")
    mmc_parser.add_argument("--service-rate", type=float, default=1.5, help="Service rate per server (mu)")
    mmc_parser.add_argument("--num-servers", type=int, default=4, help="Number of servers (c)")
    mmc_parser.add_argument("--initial-queue-length", type=int, default=0, help="Initial queue length")
    mmc_parser.add_argument("--max-capacity", type=int, default=None, help="Maximum queue capacity")
    mmc_parser.add_argument("--max-time", type=float, default=100.0, help="Maximum simulation time")
    mmc_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    mmc_parser.set_defaults(func=run_mmc_simulation)
    
    # Test parser
    test_parser = subparsers.add_parser("test", help="Run test suite")
    test_parser.set_defaults(func=run_tests)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no arguments, show help
    if args.command is None:
        parser.print_help()
        return
    
    # Execute the appropriate function
    args.func(args)

if __name__ == "__main__":
    main() 