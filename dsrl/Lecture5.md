# Lecture 5 · Model-free control

These are the notes taken during the RL Course by David Silver.

[TOC]

## Introduction

In the last lecture we studied how to estimate the value of a policy in an unknown MDP. In this lecture we will study how to find the optimal policy in an unknown MDP.

### On and off-policy learning

* __On-policy__ learning means learning while on the job
* __Off-policy__ learning is learning by seeing someone act

## Monte-Carlo control

We will be using a framework similar to the one used in policy iteration: consisting on a policy evaluation and a policy improvement steps, iteratively applied.

The naive approach would be to use MC evaluation and greedily improve the policy. This poses some problems.

### Using the Action-Value function

The greedy improvement of the policy makes use of the full structure of the MDP. We don't want to do this here.

If we cache the action-value function instead $Q(s,a)$, we don't need to take into account the structure of the MDP. The greedy policy improvement step will then be:
$$
\pi'(s) = \textrm{argmax}_{a\in A} Q(s, a)
$$

### $\epsilon$-greedy policy improvement

If we always greedily update the policy, we may not visit states that could give us a better value. This is known as the exploration-exploitation trade-off.

The simplest way to ensure exploration is the $\epsilon$-greedy.

Assume we have m actions. Then we choose the greedy action with probability $1-\epsilon$ and an action at random with probability $\epsilon$.
$$
\pi(a|s) = \begin{cases}
\epsilon/m + 1-\epsilon &\textrm{if } a^* = \textrm{argmax}_{a\in A} Q(s,a) \\
\epsilon/m &\textrm{otherwise}
\end{cases}
$$

> __Theorem:__
>
> For any $\epsilon$-greedy policy $\pi$, the $\epsilon$-greedy policy $\pi'$ with respect to $q_\pi$ is an improvement, $v_{\pi'}(s) \geq v_\pi(s)$.
>
> The proof can be found in the slides.

### GLIE

Greedy in the Limit with Infinite Exploration. Conditions that guarantee that we arrive to the optimal policy.

* All state-action pairs are explored infinitely many times: $\lim_{k\rightarrow \infty} N_k(s,a)=\infty$.
* The policy converges on a greedy policy: $\lim_{k\rightarrow \infty} \pi_k(a|s) = \mathbf{1}(a=\textrm{argmax}_{a'\in A} Q_k(s,a'))$.

$\epsilon$-greedy is GLIE if $\epsilon$ reduces to zero at $\epsilon_k=\frac{1}{k}$.

> __Theorem:__
>
> GLIE MC control converges to the optimal action-value function

## MC vs TD control

TD learning has several advantages over MC:

* Lower variance
* Online
* Incomplete sequences

Idea: use TD instead of MC in our control loop

* Apply TD to Q(S, A)
* Use $\epsilon$-greedy policy improvement
* Update at every time-step

### Sarsa

<img src="Lecture5.assets/image-20210126194400764.png" alt="image-20210126194400764" style="zoom:50%;" />
$$
Q(S,A) \leftarrow Q(S,A) + \alpha (R+\gamma Q(S', A') -Q(S,A))
$$
The idea is to update at every time-step to make use of the best estimate of $Q(S,A)$.

### Convergence of Sarsa

> __Theorem:__
>
> Sarsa converges to the optimal action-value function under the conditions:
>
> * GLIE sequence of policies $\pi_t(a|s)$
> * Robbins-Monro sequence of step-sizes $\alpha_t$
>     * $\sum_{t=1}^\infty \alpha_t = \infty$
>     * $\sum_{t=1}^\infty \alpha^2_t < \infty$

## n-step Sarsa

In the same way as in the last lecture, we extend the updates to n-step episodes.

Define the n-step Q-return:
$$
q_t^{(n)} = R_{t+1}+\gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n})
$$
The n-step Sarsa updates towards the n-step Q-return:
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t)+ \alpha \left(q_t^{(n)} - Q(S_t, A_t)\right)
$$

### Forward view Sarsa($\lambda$)

In the same way as in prediction, we weight each n-step sample by a factor $(1-\lambda) \lambda^{n-1}$.
$$
\begin{align}
	q_t^\lambda &=(1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} q_t^{(n)} \\
	Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \alpha (Q_t^\lambda - Q(S_t, A_t))
\end{align}
$$

### Backward view Sarsa($\lambda$)

By using eligibility traces we can find an equivalent algorithm to Sarsa that has a backward view instead of a forward one.
$$
E_t(s,a) = \gamma \lambda E_{t-1}(s,a) + \mathbf{1}(S_t=s, A_t=a)
$$


Updates are performed as:
$$
\delta_t =R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \\
Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t E_t(s,a)
$$

## Off-policy learning

In this case, the agent is following the behaviour policy $\mu(a|s)$, but we want to evaluate a target policy $\pi(a|s)$ to compute $v_\pi(s)$ or $q_\pi(s,a)$.

This is important because:

* we want to learn from observing other agents
* re-use experience from other policies
* learn about the optimal policy while following an exploratory policy
* learn about multiple policies while following only one

### Importance sampling

This is used to estimate the expectation of a different distribution:
$$
\begin{align}
	\mathbb{E}_{X\sim P}[f(x)] &= \sum P(X) f(X) \\
	&= \sum Q(X) \frac{P(X)}{Q(X)}f(X) \\
	&= \mathbb{E}_{X\sim Q} \left[ \frac{P(X)}{Q(X)} f(X) \right]
\end{align}
$$

### Importance sampling for off-policy MC

The approach is to multiply importance sampling corrections along the whole episode, to account for the expected return with the evaluated policy.

However, this technique has high variance which makes it useless in practice.

MC learning does not work off-policy.

### Importance sampling for off-policy TD

We use TD targets generated from $\mu$ to evaluate $\pi$.

The technique is to weight the TD target by importance sampling.
$$
V(S_t) \leftarrow V(S_t) + \alpha \left( \frac{\pi(A_t|S_t)}{\mu(A_t|S_t)} (R_{t+1} + \gamma V(S_{t+1})) - V(S_t) \right)
$$
This also increases the variance, with a chance to blow up. But it has much lower variance than MC importance sampling.

### Q-learning

__No__ importance sampling used here.

The next action is chosen using behaviour policy $A_{t+1} \sim \mu(·|S_t)$

But we also consider an alternative successor action $A' \sim \pi(·|S_t)$

We update $Q(S_t, A_t)$ towards value of alternative action:
$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha (R_{t+1} +\gamma Q(S_{t+1}, A') -Q(S_t,A_t))
$$

### Off-policy control with Q-learning

Both behaviour and target policies will improve.

The target policy $\pi$ is greedy w.r.t. $Q(s,a)$.

The behaviour policy $\mu$ is greedy w.r.t. $Q(s,a)$.

The Q-learning target simplifies:
$$
\begin{align}
	&R_{t+1} + \gamma Q(S_{t+1}, A') \\
	=& R_{t+1} + \gamma Q(S_{t+1}, \textrm{argmax}_{a'} Q(S_{t+1},a')) \\
	=& R_{t+1} + \max_{a'} \gamma Q(S_{t+1}, a')
\end{align}
$$
<img src="Lecture5.assets/image-20210127114935322.png" alt="image-20210127114935322" style="zoom:67%;" />

> __Theorem:__
>
> Q-learning control converges to the optimal action-value function.

<img src="Lecture5.assets/image-20210127115115923.png" alt="image-20210127115115923" style="zoom: 80%;" />