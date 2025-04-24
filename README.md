# Queue Reinforcement Learning (QRL)

## Overview
This project implements simulation and reinforcement learning for queueing systems, with a focus on dynamically controlling service rates and resource allocation. It provides tools for modeling M/M/1 and M/M/c queues, simulating their behavior, and applying reinforcement learning to optimize their performance.

## TODO
- [] training pipeline + script
- [] model eval script
- [] visualization pipline + interactive experiement env
- [] baselines
- [] (potentially) RLHF based real time rate tuning and adjustment models

## Mathematical Background

### Queueing Theory Fundamentals

#### Markov Processes
Queueing systems are typically modeled as continuous-time Markov chains (CTMCs). A CTMC is a stochastic process {X(t), t ≥ 0} with the Markov property:

P(X(t+s) = j | X(s) = i, X(u) = x(u), 0 ≤ u < s) = P(X(t+s) = j | X(s) = i)

for all s, t ≥ 0 and states i, j. This memoryless property is fundamental to queueing models with exponential interarrival and service times.

#### Birth-Death Processes
Birth-death processes are a special case of CTMCs where transitions only occur between adjacent states. The state typically represents the number of customers in the system.

The infinitesimal generator matrix Q of a birth-death process has the form:
```
    | -λ₀   λ₀    0     0     ... |
    | μ₁   -(λ₁+μ₁)  λ₁    0     ... |
Q = | 0     μ₂   -(λ₂+μ₂)  λ₂    ... |
    | 0     0     μ₃   -(λ₃+μ₃)  ... |
    | ...   ...   ...   ...    ... |
```

where λₙ is the birth rate (arrival rate) when in state n, and μₙ is the death rate (service rate) when in state n.

### M/M/1 Queue

An M/M/1 queue has:
- Poisson arrivals (rate λ)
- Exponential service times (rate μ)
- Single server
- First-come, first-served (FCFS) discipline

Key performance metrics for M/M/1:

- Traffic intensity (utilization): ρ = λ/μ
- Steady-state probability of n customers: π₍ₙ₎ = (1-ρ)ρⁿ for ρ < 1
- Mean number of customers: L = ρ/(1-ρ)
- Mean waiting time: W = 1/(μ-λ)
- Mean queue length: L_q = ρ²/(1-ρ)
- Mean time in queue: W_q = ρ/(μ-λ)

### M/M/c Queue

An M/M/c queue has:
- Poisson arrivals (rate λ)
- Exponential service times (rate μ per server)
- c identical servers in parallel
- FCFS discipline

Key performance metrics for M/M/c:

- Traffic intensity: ρ = λ/(cμ)
- Erlang's C formula (probability of waiting): C(c,ρ) = ((cρ)ᶜ/c!)/(∑ₖ₌₀ᶜ⁻¹(cρ)ᵏ/k! + (cρ)ᶜ/c!(1-ρ))
- Mean number of customers: L = cρ + C(c,ρ)·ρ/(1-ρ)
- Mean waiting time: W = 1/μ + C(c,ρ)/(cμ-λ)
- Mean queue length: L_q = C(c,ρ)·ρ/(1-ρ)
- Mean time in queue: W_q = C(c,ρ)/(cμ-λ)

### Gillespie Algorithm

The Gillespie algorithm (also known as the Stochastic Simulation Algorithm or SSA) is used to simulate the exact trajectory of CTMCs:

1. Calculate the total transition rate q = ∑ᵢⱼqᵢⱼ where i is the current state
2. Sample the time until the next event from an exponential distribution with rate q: Δt ~ Exp(q)
3. Sample the next state j with probability qᵢⱼ/q
4. Update the time t = t + Δt and state i = j
5. Repeat until the end condition is met

## Reinforcement Learning for Queue Control

### Markov Decision Process Formulation

Queue control can be formulated as a Markov Decision Process (MDP):
- State space: Queue length, server states, etc.
- Action space: Service rate adjustment, number of servers, etc.
- Transition function: Defined by the queueing dynamics
- Reward function: Typically a combination of throughput, waiting time, and resource costs

### Value Functions and Bellman Equations

The state-value function V(s) represents the expected return starting from state s:

V(s) = 𝔼[∑ᵏ₌₀^∞ γᵏ·rₖ₊₁ | S₀ = s]

The action-value function Q(s,a) represents the expected return after taking action a in state s:

Q(s,a) = 𝔼[∑ᵏ₌₀^∞ γᵏ·rₖ₊₁ | S₀ = s, A₀ = a]

Bellman equations relate the value of a state to the values of its successor states:

V(s) = ∑ₐ π(a|s) ∑ₛ′ p(s′|s,a)[r(s,a,s′) + γV(s′)]
Q(s,a) = ∑ₛ′ p(s′|s,a)[r(s,a,s′) + γ∑ₐ′ π(a′|s′)Q(s′,a′)]

### Reinforcement Learning Algorithms

Several RL algorithms can be applied to queue control:

1. **Q-learning**: Model-free algorithm that learns the action-value function:
   Q(s,a) ← Q(s,a) + α[r + γ·max_a′ Q(s′,a′) - Q(s,a)]

2. **SARSA**: On-policy algorithm that updates based on the next action taken:
   Q(s,a) ← Q(s,a) + α[r + γ·Q(s′,a′) - Q(s,a)]

3. **Policy Gradient**: Directly optimizes the policy parameters θ:
   θ ← θ + α·∇log π(a|s;θ)·G_t

4. **Actor-Critic**: Combines value function approximation and policy optimization:
   θ ← θ + α·∇log π(a|s;θ)·δ
   where δ = r + γ·V(s′) - V(s) is the temporal difference error

## Optimization Objectives

Common optimization objectives in queue control include:

1. **Minimizing average waiting time**: min E[W]
2. **Maximizing throughput**: max λ_eff (effective arrival rate)
3. **Minimizing resource costs**: min c·μ (service cost) or min n·c_s (server cost)
4. **Multi-objective optimization**: min w₁·E[W] + w₂·cost, where w₁ and w₂ are weights

## Example Policies

1. **Threshold Policy**: Increase service rate when queue length exceeds a threshold
   μ(t) = μ₁ if q(t) ≤ K, μ₂ otherwise (μ₂ > μ₁)

2. **Linear Control Policy**: Service rate proportional to queue length
   μ(t) = μ₀ + α·q(t)

3. **N-Policy**: Only serve when queue length reaches N customers
   serve if q(t) ≥ N, idle otherwise

4. **Dynamic Server Allocation**: Adjust number of servers based on queue length
   c(t) = min(max(⌈q(t)/K⌉, c_min), c_max)

## References

1. Kleinrock, L. (1975). Queueing Systems, Volume 1: Theory.
2. Tijms, H.C. (2003). A First Course in Stochastic Models.
3. Sutton, R.S., & Barto, A.G. (2018). Reinforcement Learning: An Introduction.
4. Puterman, M.L. (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming.
5. Gillespie, D.T. (1977). Exact stochastic simulation of coupled chemical reactions.

## Installation and Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run M/M/1 queue simulation
python src/main.py mm1 --arrival-rate 2.0 --service-rate 3.0

# Run M/M/c queue simulation
python src/main.py mmc --arrival-rate 5.0 --service-rate 1.5 --num-servers 4

# Run tests
python src/main.py test
``` 
