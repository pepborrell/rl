# Lecture 1 · Introduction

These are the notes taken during the RL Course by David Silver.

[TOC]

**Reinforcement Learning** (RL) is essentially the science of decision making.

__Characteristics__ that make it different to other machine learning fields (mainly supervised learning):

* There is no supervisor
* Feedback is delayed, not at the time of the decision
* Time matters, data is sequential
* The agent influences the environment by its decisions

## The RL problem

### Rewards

A reward $R_t$ is a scalar feedback signal. It serves as the way to show the agent how well it is doing.

The goal of the agent is to pick up actions so it maximises the cumulative reward. This is called the __reward hypothesis__.

#### Examples:

* Controlling a stunt helicopter: positive rewards if the desired trajectory is followed. Negative rewards for crashing.
* In Backgammon: positive or negative rewards for winning or losing a game.

### Sequential decision making

Actions may have long term consequences.

The reward may be delayed.

In some cases, a short-term loss might be a large long-term gain.

### Agent and environment

The agent sees an observation of the environment, a reward and takes an action based on what it sees.

The environment is influenced by the actions taken by the agent. It then emits an observation and a reward to the agent.

### History and state

The __history__ is the sequence of observations, actions and rewards. It contains the observable variables up to a certain moment.

The __state__ is the information used to determine what action is taken. It formally is a function of the environment. It may be only the last information, for example.

### Environment state

The environment's own private representation. The data the environment chooses to pick the next observation or reward.

It is not usually visible to the agent.

### Agent state

It's the agent's own internal representation of the states. It is the information that is useful for the agent.

### Information state

Also called Markov state, it contains all useful information from the history.

We say that a state $S_t$ is __Markov__ $\iff \Pr[S_{t+1}|S_t] = \Pr[S_{t+1}|S_1, \dots, S_t]$. "The future is independent on the past given the present".

Once we know this state, we can throw away the history. We can also say that the state is a sufficient statistic of the future.

The environment state $S_t^e$ is Markov, because it is the information the environment uses to determine what will happen next.

The history $H_t$ is Markov, by definition of a Markov state.

### Fully observable environments

It is the situation where the agent sees the environment state.

The formalism to work with these environments is Markov decision process.

### Partially observable environments

The situation where the agent indirectly observes the environment.

The formalism is called partially observable Markov decision process (POMDP).

The agent must construct its own state representation.

## Main components of an RL agent

### Policy

The function that decides what action to take. It basically is the agent's behaviour.

__Notation:__ $a = \pi(s)$ (deterministic) or $a=\pi(s|a)$ (stochastic).

### Value function

It tells how good a state or action is. It's the expected future total reward.

It depends on the policy.

__Notation:__ $v_\pi(s) = \mathbb{E}_\pi [R_t+\sigma R_{t+1} + \sigma^2 R_{t+2} + \cdots|S_t=s]$.

### Model

The representation of the environment. The model predicts what the environment will do (either the next state or the next reward).

### Taxonomy of RL agents

#### Value/Policy distinction

* __Value based__: it has a value function and has no policy (or an implicit one).
* __Policy based__: it has an explicit policy and no value function.
* __Actor critic__: it has both policy and value function.

#### Model distinction

* __Model free__: policy and/or value function, without model.
* __Model based__: policy and/or value function, with a model.

## Subproblems in RL

__The Reinforcement Learning problem__: the environment is unknown. The agent improves its policy by interacting with the environment.

__The Planning problem__: a model is known, the agent improves its policy by computing outcomes.

These problems are interlinked. If we know the environment, we can plan. But we need to know how the environment looks like first.

**Exploration and exploitation**: in some cases we may need to give up on some reward we know of to get new information about the environment.

We need to know the difference between **prediction and control**. **Prediction** means to predict the reward we will get based on a policy. **Control** is determining the optimal behaviour, computing the optimal policy.