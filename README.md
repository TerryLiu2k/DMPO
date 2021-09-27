Author: Hangrui (Henry) Bi

Algorithms:
1. DMPPO
2. DPPO
* radius and radius_q: 
    * The observable radius of p should be 1.
    * The V of each agent predicts the local reward. (Alternatively, we may sum the k-hop local Qs, but that cannot be elegantly implemented for discrete action multiagent and requires computing Q multiple times to get the derivative against the action of each agent)
    
Environments:
CACC Catchup, Slowdown.
Ring Attenuation.
Figure Eight.


Interface:
The state, reward, done should all be numpy arrays
For MARL, done should be given for each agent, although we assume they are the same.

Usage:
```python
python launcher_ppo.py
```