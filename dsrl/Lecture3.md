# Lecture 3 · Planning by Dynamic Programming

These are the notes taken during the RL Course by David Silver.

[TOC]

## Introduction to dynamic programming

The word dynamic comes from the fact that the problem has some sequential component.

__Dynamic programming__ solves problems by breaking them into subproblems and soving them.

Dynamic programming is a great fit for problems which have __two properties__:

* Optimal substructure: the principle of optimality applies. This principle says that a problem can be broken down into two or more parts that tell me how to solve the higher level problem.
* Overlapping subproblems: this happens when subproblems recur many times: the solutions can be reused for other subproblems.

MDPs satisfy both of these properties. The Bellman equation gives recursive decomposition, and the value function can be cached and reused.

Dynamic programming is used for planning in an MDP. We can plan for either prediction (the value function given a policy) and for control (to get the optimal policy and optimal value function).

## Policy evaluation

In this problem we want to evaluate a policy $\pi$. We will do so by iteratively applying the Bellman equation to progressively get to a more accurate value function.

Synchronous backups:

```
At each iteration k+1:
	For all states s:
	Update v[k+1](s) from v[k](s')
		(where s' is a successor state of s)
```

By having the true value function, we can get a better policy by greedily building it.

## Policy iteration

The process done here builds upon policy evaluation.

Given a policy $\pi$, we evaluate the policy to get $v_\pi$ and then we greedily build a new policy $\pi'$ from the value function.

![Elucidating Policy Iteration in Reinforcement Learning — Jack's Car Rental  Problem | by Aditya Rastogi | Towards Data Science](https://miro.medium.com/max/2624/1*udhphWhqjadT-osAQhL6AQ.png)

By iteratively doing this process we can guarantee that policy iteration always converges to $\pi^*$.

A more rigorous explanation about the convergence of this algorithm is given in the video and will not be written down here. It is based on the fact that a greedy action at a given point with respect to a value function is better than or equal to following the defined policy.

When improvements stop, we satisfy the Bellman optimality equation and thus have reached the optimal policy.

### Modified policy iteration

By experimentally applying policy iteration, it is found that we do not need the value function to converge in order to get a better policy (or the same policy we would find if it had converged).

The modification is mainly stopping before the value function converges. Usually, the stopping condition is to stop after $k$ iterations.

It can be proved that this approach converges to the optimal policy as well.

## Value iteration

Intuitively, we can divide an optimal policy into:

* An optimal first action $A_*$
* An optimal policy from the successor state $S'$

### Theorem (Principle of Optimality)

> A policy $\pi(a|s)$ achieves the optimal value from state $s$, $v_\pi(s)=v_*(s)$ if and only if:
>
> For any state $s'$ reachable form $s$, $\pi$ achieves the optimal value from state $s'$, $v_\pi(s')=v_*(s')$.

### Intuition behind the algorithm

If we know the solution to subproblems $v_*(s')$, the solution $v_*(s)$ can be found by performing a one-step lookahead using the known subproblems (using the Bellman optimality equation).

Value iteration aims to apply these updates iteratively.

The intuition is to start with the final rewards and go backwards. Actually, the algorithm loops over all states, without taking into account the structure of the problem.

It works with loops and stochastic MDPs.

### The algorithm

The algorithm works by iteratively applying the Bellman optimality equation.
$$
\mathbf{v}_{k+1}=\max_{a\in A}[\mathbf{R^a} + \gamma \mathbf{P^a} \mathbf{v}_k]
$$


By using synchronous backups we update the value function.

There is no explicit policy, but this process is the same as performing modified policy iteration with $k=1$. This is because we are _greedily_ selecting the actions we take to achieve the maximum possible value at the next state.

## Classification of dynamic programming algorithms

### Prediction

The equation we use to do prediction is the Bellman expectation equation. The algorithm covered in this class is Iterative Policy Evaluation.

### Control

We have seen two algorithms to do control (or finding the optimal policy).

* The policy iteration algorithm uses the Bellman expectation equation and a greedy policy improvement.
* The value iteration algorithm uses the Bellman optimality equation.

### Complexity

The algorithms based on the state-value function (like the ones covered in this lecture) have a complexity of $O(mn^2)$ per iteration for m actions and n states.

The algorithms based on the action-value function have a complexity of $O(m^2n^2)$ per iteration. These algorithms will be covered in the next lectures (model-free control).

## Advanced DP techniques

* Using **asynchronous** DP can significantly reduce computation. If all states continue to be selected, this method still converges. Three ideas to asynchronously perform DP:
    * __In-place DP__: only storing one copy of the value function, not distinguishing between iteration steps. We always use the latest version to update the value.
    * __Prioritised sweeping__: the ordering of updates matters. We update states based on the Bellman error : the largest the error, the sooner we update it.
    * __Real-time DP__: we select the states the agent actually visits. We collect samples from the agent and update using those samples.

* DP uses __full-width backups__. This makes backups extremely expensive. In subsequent lectures we will sample backups, instead of taking into account the full branching factor.

## Contraction mapping

The contraction mapping theorem explains how the algorithms shown converge, the uniqueness of their solutions and the speed at which they converge. This theorem is explained in the lecture notes.