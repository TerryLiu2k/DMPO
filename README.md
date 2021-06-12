Author: Hangrui (Henry) Bi

Algorithms: (only discrete action supported now)
1. Qlearning
2. SAC
3. MBPO 
4. DMPO: Hyperparameters, design and implementation detail
* radius and radius_q: 
    * The observable radius of p should be 1, if the MDP is decomposable. While I assume the radius of pi can be larger.
    * The Q of each agent predicts the k-hop mean reward. (Alternatively, we may sum the k-hop local Qs, but that cannot be elegantly implemented for discrete action multiagent and requires computing Q multiple times to get the derivative against the action of each agent)
    
Environments:
Gym Envs
Prisoner's Dilemma
CACC-catchup and CACC-slowdown
* SAC, one update 10 steps (1 update per step is stable, but computationally slower), bs 256 per sample 
* Slowdown, reward must be clipped, besides rescaled
    The reward of our method oscillates between -400 to -1000, which is comparable to the state-of-the-art
* Catchup
    This can be well-solved without cooperation at all
    Using SAC, target entropy 0.2, radius_q = 3, our cooperative version may get stuck in local optimum: the leading car never accelerate.
ASTC-Grid, ATSC-RealNet


Features:
* Object Oriented Ontology
* Multiagent RL for Networked Control Systems
    * Supports communication with extended action space
* Advanced logging
    * Logger with a local buffer and a remote hosted dashboard
    * Easy to control log period (wall time)
    * Logger hierarchy scaling to multiagent and multiprocess environments
    * Multiprocessing
* Visualization of Environments
* Factorized discrete action space
* Multiprocessing
    * The agents run in parallel in their own process
* Scalability
    * Supports using both CPU and GPU
    * Supports both multiprocessing parallelism (based on ray) and sequential execution for multiagent
* Easy Debugging and Profiling    
    
* Factorized discrete action space
    agent_args.action_space = [n1, n2...]

    
    

Interface:
The state, reward, done should all be numpy arrays
For MARL, done should be given for each agent, although we assume they are the same

Usage:
```python
python launcher.py
```