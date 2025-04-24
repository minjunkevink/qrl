"""
Visualization utilities for stochastic processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import List, Tuple, Dict, Any, Optional, Union
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd
from pathlib import Path
import os

def plot_trajectory(
    history: List[Tuple[float, Any]], 
    title: str = "State Trajectory", 
    xlabel: str = "Time", 
    ylabel: str = "State",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    color: str = 'blue',
    linestyle: str = '-',
    marker: Optional[str] = None,
    alpha: float = 1.0,
    grid: bool = True
):
    """
    Plot a trajectory of a stochastic process.
    
    Args:
        history: List of (time, state) tuples
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        figsize: Figure size (width, height) in inches
        save_path: Path to save the figure (if None, figure is not saved)
        show: Whether to display the plot
        color: Line color
        linestyle: Line style
        marker: Marker style (if None, no markers)
        alpha: Transparency level
        grid: Whether to show grid lines
    """
    if not history:
        raise ValueError("History is empty")
    
    # Extract time and state values
    times, states = zip(*history)
    
    # Convert states to numeric values if possible
    try:
        states = [float(state) for state in states]
    except (TypeError, ValueError):
        # If states cannot be converted to float, keep as is
        pass
    
    plt.figure(figsize=figsize)
    plt.plot(times, states, color=color, linestyle=linestyle, marker=marker, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid:
        plt.grid(True, alpha=0.3)
    
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_multiple_trajectories(
    histories: List[List[Tuple[float, Any]]],
    labels: Optional[List[str]] = None,
    title: str = "State Trajectories",
    xlabel: str = "Time",
    ylabel: str = "State",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    alpha: float = 0.8,
    grid: bool = True
):
    """
    Plot multiple trajectories on the same graph.
    
    Args:
        histories: List of history lists, each containing (time, state) tuples
        labels: Labels for each trajectory (for legend)
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        figsize: Figure size (width, height) in inches
        save_path: Path to save the figure (if None, figure is not saved)
        show: Whether to display the plot
        colors: List of colors for each trajectory
        linestyles: List of line styles for each trajectory
        markers: List of marker styles for each trajectory
        alpha: Transparency level
        grid: Whether to show grid lines
    """
    if not histories:
        raise ValueError("No histories provided")
    
    plt.figure(figsize=figsize)
    
    # Default parameters
    if colors is None:
        # Create a colormap with distinct colors
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(histories))]
    
    if linestyles is None:
        linestyles = ['-'] * len(histories)
    
    if markers is None:
        markers = [None] * len(histories)
    
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(len(histories))]
    
    # Plot each trajectory
    for i, history in enumerate(histories):
        if not history:
            continue
            
        times, states = zip(*history)
        
        # Convert states to numeric values if possible
        try:
            states = [float(state) for state in states]
        except (TypeError, ValueError):
            # If states cannot be converted to float, keep as is
            pass
        
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        label = labels[i % len(labels)]
        
        plt.plot(times, states, color=color, linestyle=linestyle, 
                 marker=marker, alpha=alpha, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid:
        plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def animate_trajectory(
    history: List[Tuple[float, Any]],
    title: str = "State Trajectory Animation",
    xlabel: str = "Time",
    ylabel: str = "State",
    figsize: Tuple[int, int] = (10, 6),
    interval: int = 50,  # milliseconds between frames
    save_path: Optional[str] = None,
    color: str = 'blue',
    trail_length: Optional[int] = None,  # Number of past points to show
    grid: bool = True
) -> animation.FuncAnimation:
    """
    Create an animation of a stochastic process trajectory.
    
    Args:
        history: List of (time, state) tuples
        title: Animation title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        figsize: Figure size (width, height) in inches
        interval: Time between animation frames in milliseconds
        save_path: Path to save the animation (if None, animation is not saved)
        color: Line color
        trail_length: Number of past points to show (if None, show all)
        grid: Whether to show grid lines
        
    Returns:
        Matplotlib animation object
    """
    if not history:
        raise ValueError("History is empty")
    
    # Extract time and state values
    times, states = zip(*history)
    
    # Convert states to numeric values if possible
    try:
        states = [float(state) for state in states]
    except (TypeError, ValueError):
        # If states cannot be converted to float, keep as is
        pass
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot([], [], color=color, marker='o', markersize=4)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set axis limits with some padding
    time_padding = (max(times) - min(times)) * 0.05
    state_min = min(states)
    state_max = max(states)
    state_padding = (state_max - state_min) * 0.1 if state_max > state_min else 1.0
    
    ax.set_xlim(min(times) - time_padding, max(times) + time_padding)
    ax.set_ylim(state_min - state_padding, state_max + state_padding)
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    # Initialization function for animation
    def init():
        line.set_data([], [])
        return (line,)
    
    # Animation function
    def animate(i):
        if trail_length is None:
            # Show all past points
            x = times[:i+1]
            y = states[:i+1]
        else:
            # Show only a trail of points
            start_idx = max(0, i - trail_length + 1)
            x = times[start_idx:i+1]
            y = states[start_idx:i+1]
        
        line.set_data(x, y)
        return (line,)
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(times), interval=interval, blit=True
    )
    
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Determine file format from extension
        extension = Path(save_path).suffix.lower()
        if extension == '.gif':
            anim.save(save_path, writer='pillow', fps=1000/interval)
        elif extension in ['.mp4', '.mov', '.avi']:
            anim.save(save_path, writer='ffmpeg', fps=1000/interval)
        else:
            raise ValueError(f"Unsupported animation format: {extension}")
    
    return anim

def plot_distribution(
    states: List[Any],
    title: str = "State Distribution",
    xlabel: str = "State",
    ylabel: str = "Frequency",
    figsize: Tuple[int, int] = (10, 6),
    bins: Union[int, str, List[float]] = 'auto',
    color: str = 'blue',
    alpha: float = 0.7,
    save_path: Optional[str] = None,
    show: bool = True,
    kde: bool = False,
    grid: bool = True
):
    """
    Plot the distribution of states.
    
    Args:
        states: List of state values
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        figsize: Figure size (width, height) in inches
        bins: Histogram bins specification
        color: Bar color
        alpha: Transparency level
        save_path: Path to save the figure (if None, figure is not saved)
        show: Whether to display the plot
        kde: Whether to overlay a kernel density estimate
        grid: Whether to show grid lines
    """
    if not states:
        raise ValueError("States list is empty")
    
    plt.figure(figsize=figsize)
    
    # Check if states are numeric
    numeric_states = True
    try:
        [float(state) for state in states]
    except (TypeError, ValueError):
        numeric_states = False
    
    if numeric_states:
        # For numeric states, use a histogram
        n, bins, patches = plt.hist(states, bins=bins, color=color, 
                                    alpha=alpha, density=kde)
        
        if kde:
            # Add kernel density estimate
            from scipy import stats
            density = stats.gaussian_kde(states)
            x = np.linspace(min(states), max(states), 1000)
            plt.plot(x, density(x), 'r-', alpha=0.7)
    else:
        # For non-numeric states, use a bar chart
        state_counts = {}
        for state in states:
            state_str = str(state)
            state_counts[state_str] = state_counts.get(state_str, 0) + 1
        
        plt.bar(state_counts.keys(), state_counts.values(), color=color, alpha=alpha)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid:
        plt.grid(True, alpha=0.3)
    
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_heatmap(
    data: Union[np.ndarray, List[List[float]]],
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Heatmap",
    xlabel: str = "X",
    ylabel: str = "Y",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    show_values: bool = True,
    value_format: str = '{:.2f}',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot a heatmap.
    
    Args:
        data: 2D array of values
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        figsize: Figure size (width, height) in inches
        cmap: Colormap name
        show_values: Whether to show values in cells
        value_format: Format string for cell values
        save_path: Path to save the figure (if None, figure is not saved)
        show: Whether to display the plot
    """
    # Convert to numpy array if needed
    data_array = np.array(data)
    
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = ax.imshow(data_array, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set tick labels
    if x_labels:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        ax.set_xticks(np.arange(data_array.shape[1]))
        
    if y_labels:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticks(np.arange(data_array.shape[0]))
    
    # Add grid lines
    ax.set_xticks(np.arange(data_array.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data_array.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    
    # Show values in cells
    if show_values:
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                value = data_array[i, j]
                text_color = 'white' if value > (data_array.max() + data_array.min()) / 2 else 'black'
                ax.text(j, i, value_format.format(value), 
                        ha="center", va="center", color=text_color)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig, ax 