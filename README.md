# Homework 2 for CS 690S AI Alignment

### Overview

In this assignment, you will be implementing maximum entropy inverse reinforcement learning, as described by Ziebart et al. (2008) in``Maximum Entropy Inverse Reinforcement Learning".

**NOTE: There are two typos in Algorithm 1 of that paper:**
- In step 4, instead of D_{s_i, t}, it should be: 
  - ![image](https://github.com/sniekum/MaxEntIRL_assignment/assets/1664131/e7c49e02-84b1-4229-b6b4-9b27812aedb6)

- In step 5, it should be:
  - ![image](https://github.com/sniekum/MaxEntIRL_assignment/assets/1664131/978ffc22-f707-490e-a754-fb340a916908)
  - In other words, the probability of a state s_i at the next timestep is based on how often you started in all the states s_k that you could have come from, multiplied by the appropriate action and transition probabilities that would have taken you from s_k to s_i

Your task is to write three functions, as specified in the provided file.  The three functions are:

1. **calcMaxEntPolicy(trans\_mat, horizon, r\_weights, state\_features):**
  
   - Description: For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal return trajectories
  
   - Parameters:
     - trans_mat: an S x A x S' array of transition probabilities from state s to s' if action a is taken
     - horizon: the finite time horizon (int) of the problem for calculating state frequencies
     - r_weights: a size F array of the weights of the current reward function to evaluate
     - item state_features: an S x F array that lists F feature values for each state in S
     - item term_index: the index of the special terminal state
  
   - Return: an S x A policy in which each entry is the probability of taking action a in state s


2. **calcExpectedStateFreq(trans\_mat, horizon, start\_dist, policy):**
  
   - Description: Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon
  
   - Parameters:
     - trans_mat: an S x A x S' array of transition probabilities from state s to s' if action a is taken
     - horizon: the finite time horizon (int) of the problem for calculating state frequencies
     - start_dist: a size S array of starting start probabilities - must sum to 1
     - policy: an S x A array array of probabilities of taking action a when in state s
  
    - Return: a size S array of expected state visitation frequencies


3. **maxEntIRL(trans\_mat, state\_features, demos, seed\_weights, n\_epochs, horizon, learning\_rate):**
  
   - Description: Compute a MaxEnt reward function from demonstration trajectories.  This will invoke the above two functions.

   - Parameters: 
     - trans_mat: an S x A x S' array that describes transition probabilities from state s to s' if action a is taken
     - state_features: an S x F array that lists F feature values for each state in S
     - demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
     - seed_weights: a size F array of starting reward weights
     - n\_epochs: how many times (int) to perform gradient descent steps
     - horizon: the finite time horizon (int) of the problem for calculating state frequencies
     - learning_rate: a multiplicative factor (float) that determines gradient step size
     - term_index: the index of the special terminal state

   - Return: a size F array of reward weights


The transition matrix for a 5x5 gridworld domain is provided in the Python starter code, with states laid out as follows:

```
0  1  2  3  4
5  6  7  8  9
10 11 12 13 14
15 16 17 18 19
20 21 22 23 24
```


24 is a goal state that always transitions to a special terminal state, 25.  There are 4 actions: up, down, left, and right, which move the agent accordingly (and deterministically), unless the agent would move past the edge of the world, in which it stays in place.  All actions at state 24 lead to state 25 deterministically.  All actions at state 25 lead to each possible next state with probability zero, making it terminal. 

There are 4 features and only one is active at any given state, represented 1-hot vector at each state, with the layout as follows:
```
0  0  0  0  0 
0  1  1  1  1 
0  0  2  0  0 
0  0  0  0  0 
0  0  0  0  3 
```
The special terminal state (25) has all zero state features.

**Notes:**

- $\tilde{\textrm{f}}$ should be averaged over all demonstrations trajectories (i.e. it is an expected value that needs to be divided by the number of trajectories).  The paper isn't totally clear about this.
- For all questions, seed the reward weights with all zeros unless instructed otherwise.


## Answer the following questions:

<strong>You will need to type up your responses to the following parts of the homework (preferably typeset in LaTeX) and submit your responses and code via Gradescope. You are encouraged to talk about the homework with other students and share resources, but please do not share or copy code. </strong>

1. Experiment and choose a good static step size / learning rate for gradient descent (i.e. a scalar value to multiply the gradient by, in order to decide on the magnitude of the weight update at each iteration). Describe how you chose an appropriate learning rate. 

2. Run MaxEntIRL for 100 epochs with a horizon of 15 and describe the shape of the reward function.  What does it appear to be trying to do, qualitatively, and why?  Do you think it treats state 12 in a sensible way based on the given demostrations, and might it be wrong (with respect to the ground truth reward function that generated the given optimal demonstrations)? Include a graph of the reward function in your writeup.

3. We will now do a simplified calculation of the gradient of the loss function:
   <img width="601" alt="grad" src="https://github.com/sniekum/MaxEntIRL_assignment/assets/1664131/5395c12f-17ef-4f2f-8482-0f66a2075c15">

   If you are having difficulty, you might want to refresh your memory on how to take derivatives of logs, exponentials, and how to use the chain and product rule. I strongly recommend typesetting your derivation in LaTeX (along with the rest of the writeup).

4. Compare equations 5 and 7.  Intuitively, why does the MaxEnt policy of equation 5 equally weight paths in the example in Figure 2, while the action based distribution model of equation 7 weights them differently?  Why is the weighting of equation 7 problematic?

5. Explain in your own words what $Z_{a_{ij}}$ and $Z_{s_i}$ represent over a particular horizon.

6. Why is 1 added to $Z_s$ for terminal states?  (hint: think of the equation for $Z_{s}$ in terms of returns of trajectories)

## Submission
Prepare a PDF report with your answers to the questions (preferably typeset in LaTeX) and submit the PDF along with your code in a zip file. Submit via Gradescope.

