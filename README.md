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

SUMO version:
The commit number of SUMO, available at https://github.com/eclipse/sumo used to run the results is “1d4338ab80”.

The environment specification is in environment.yml.
```
sudo apt-get install cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig
cd <sumo_dir> # please insert the correct directory name here
export SUMO_HOME="$PWD"
mkdir build/cmake-build && cd build/cmake-build
cmake ../..
make -j$(nproc)
```

Usage:
```python
python launcher_ppo.py
```


