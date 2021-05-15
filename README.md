Author: Hangrui (Henry) Bi

Algorithms:
1. Qlearning
2. SAC for discrete action
3. MBPO for discrete action

Components:
* An generic RL algorithm
* Agents: 
        QLearning
        SAC
        MBPO
        MultiAgent: a wrapper class that contains multiple agents
* Modules
        p, q, pi
* Networks
    
Features:
* Object oriented ontology
* Multiagent RL
* Advanced logging
    * Logger with a local buffer and a remote hosted dashboard
    * Easy to control log period (wall time)
    * Logger hierarchy scaling to multiagent and multiprocess environments
* Visualization of environments

Interface:
The state, reward, done should all be numpy arrays

Example:
> python -c "import algorithms.config.SAC_CartPole"
