# Lecture 2 · Markov Decision Processes

These are the notes taken during the RL Course by David Silver.

[TOC]

## Introduction

A Markov decision process formally describes an environment for RL.

In MDPs, the environment is fully observable: the current state completely characterises the process (Markov property).

Some observations:

* There are continuous MDPs: dealt with in optimal control.
* Partially observable environments can be converted into MDPs.

##  Markov processes

Characterised by the __State transition matrix__: a matrix that contains the state transition probability: $P_{ss'}=\Pr[S_{t+1}=s'|S_t=s]$.
$$
P= \textrm{from} 
\left[
\begin{matrix}
	P_{11} & \cdots & P_{1n} \\
	\vdots & \ddots & \vdots \\
	P_{n1} & \cdots & P_{nn}
\end{matrix}
\right]
$$
Each row sums up to 1.

A Markov process is a sequence of random states with the Markov property.

## Markov reward processes

A Markov reward process is a Markov chain with values.

It is a Markov chain with a reward $R$, represented by a __reward function__ $R_s = \mathbb{E}[R_t|S_t=s]$ and a discount factor $\gamma$.

We care about the **return** $G_t$.
$$
G_t = R_{t+1} + \gamma R_{t+1} + \cdots = \sum_{k=0}^\infty = \gamma^k R_{t+k+1}
$$
The discount $\gamma \in [0,1]$ is the present value of future rewards.

It is worth noting that the definition of the return does not contain any expectation operator because it represents a random sample.

The **value function** gives the long-term value of state $s$. It is the magnitude we truly care about.
$$
v(s) = \mathbb{E}[G_t|S_t=s]
$$

### Bellman equation for MRPs

By decomposing the value function into the immediate reward and the discounted value of the next state, we obtain the Bellman equation.
$$
\begin{align}
    v(s) &= \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots | S_t=s] \\
    &= \mathbb{E}[R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \cdots) | S_t=s] \\
    &= \mathbb{E}[R_{t+1} + \gamma G_t | S_t=s] \\
    &= \mathbb{E}[R_{t+1} + \gamma v(S_{t+1}) | S_t=s]
\end{align}
$$
This equation can be expressed using matrices: $\mathbf{v} = \mathbf{R} + \gamma P \mathbf{v}$, where $P$ is the transition probability matrix.

Because this equation is linear, we can solve it directly: $\mathbf{v} = (I - \gamma P)^{-1} \mathbf{R}$. The complexity is $O(n^3)$ for n states, which makes it an inefficient way of solving the problem.

## Markov decision processes

A Markov decision process is a Markov reward process with decisions. We include the set $A$, which is a finite set of actions.

The transition probability matrix is now $P_{ss'}^a = \Pr[S_{t+1}=s'|S_t=s, A_t = a]$. The reward function is $R_s^a = \mathbb{E}[R_t|S_t=s, A_t=a]$.

### Policies

A policy $\pi$ is a distribution over actions given states: $\pi (a|s) = \Pr[A_t=a|S_t=s]$.

A policy fully defines the behaviour of an agent.

Using the Markov property, we can make the policies dependent only on the current state.

Given an MDP and a policy, the state sequence is a Markov process. Additionally, the state and reward process is an MRP. Both are defined by the probabilities and rewards computed by the expected value using the policy.

### Value function

The __state-value function__ $v_\pi(s)$ is $v_\pi(s) = \mathbb{E}_\pi[G_t|S_t=s]$.

The __action-value function__ is $q_\pi(s,a) = \mathbb{E}_\pi[G_t|S_t=s, A_t = a]$.

### Bellman expectation equation

Decomposing the previous equation in the same way as before, we get:
$$
\begin{align}
	v_\pi(s) &= \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t=s] \\
	q_\pi(s,a) &= \mathbb{E}[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) | S_t=s, A_t=a]
\end{align}
$$
By iterating the expectations from state-value functions to action-value functions and vice-versa, we obtain the recursive Bellman expectation equations.
$$
\mathbf{v}_\pi = \mathbf{R}^\pi + \gamma P^\pi \mathbf{v}_\pi \\
\dots
$$

## Optimal value function

The __optimal state-value function__ $v_*(s)$ is the maximum value function over all policies. $v_*(s) = \max_\pi v_\pi(s)$.

The __optimal action-value function__ $q_*(s,a)$ is the maximum action-value function over all policies. $q_*(s, a) = \max_\pi q_\pi(s, a)$.

We say that an MDP is solved when we find $q_*$ for this MDP.

We need to define a partial ordering over policies: $\pi \geq \pi' \; \textrm{if} \; V_\pi(s) \geq v_{\pi'}(s), \forall s$.

### Theorem (existence of optimal policies)

For any MDP:

* There exists an optimal policy $\pi_*$, better than or equal to all other policies.
* All optimal policies achieve the optimal value function $v_*(s)$.
* All optimal policies achieve the optimal action-value function $q_*(s,a)$.

### Finding an optimal policy

An optimal policy can be found by using the actions that maximise the optimal action-value function.

There always exists an optimal deterministic policy.

### Bellman optimality equation

The optimal value functions are recursively related by the Bellman optimality equations.
$$
\begin{align}
	v_*(s) &= \max_a q_*(s,a) \\
	q_*(s,a) &= R_s^a  + \gamma \sum_{s'\in S} P_{ss'}^a v_*(s') \\
	\Rightarrow v_*(s) &= \max_a R_s^a  + \gamma \sum_{s'\in S} P_{ss'}^a v_*(s') \\
	\Rightarrow q_*(s,a) &= R_s^a  + \gamma \sum_{s'\in S} P_{ss'}^a \max_{a'} q_*(s',a') \\
\end{align}
$$
These equations are non-linear with no closed solution. However, we have several iterative solution methods to solve them.

## Extensions to MDPs

There are several extensions to the MDP framework, covered in the online slides but not explained in the video. Some examples of them are:

* Infinite and continuous MDPs.
* Partially observable MDPs.
* Undiscounted, average reward MDPs.