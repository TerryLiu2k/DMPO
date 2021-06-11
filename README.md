Author: Hangrui (Henry) Bi

Algorithms: (only discrete action supported now)
1. Qlearning
2. SAC
3. MBPO 
4. DMPO: Hyperparameters, design and implementation detail
* radius and radius_q: 
    * The observable radius of p should be 1, if the MDP is decomposable.
    * For q, the observable radius should be at least as large as the reduce radius. 
        For continous action, all actions are inputs of Q, it is easy to compute $dQ_i/da_j$.
        For discrete action? It is non-trivial, I think the best way is to compute Q multiple times.
    
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

Usage:
```python
python launcher.py
```