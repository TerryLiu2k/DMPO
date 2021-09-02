import gym
from copy import deepcopy
from gym.envs.registration import register

from ..envs.flow.envs.multiagent import MultiTrafficLightGridPOEnv
from ..envs.flow.networks import TrafficLightGridNetwork
from ..envs.flow.core import rewards
from ..envs.flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, TrafficLightParams
from ..envs.flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from ..envs.flow.controllers import SimCarFollowingController, GridRouter, RLController, IDMController, ContinuousRouter
from ray.tune.registry import register_env
from ..envs.flow.utils.registry import make_create_env

import algorithms.envs.flow.envs
from ..envs.flow.benchmarks.grid0 import flow_params as grid_flow_params

import numpy as np



class FlowGridWrapper(MultiTrafficLightGridPOEnv):
    def __init__(self, env_params, sim_params, network, simulator):
        super().__init__(env_params, sim_params, network, simulator=simulator)
        self.n_s_ls, self.n_a_ls, self.coop_gamma, self.distance_mask, self.neighbor_mask \
            = [], [], -1, np.zeros((9, 9)), np.zeros((9, 9))
        self.init_neighbor_mask()
        self.init_distance_mask()
        self.n_s_ls = [42] * 9
        self.n_a_ls = [2] * 9

    def get_state_(self):
        state = super().get_state()
        state = list(state.values())
        if not hasattr(self, 'keys') or self.keys is None:
            if isinstance(state, dict):
                self.keys = list(state.keys())
            else:
                self.keys = list(super().get_state().keys())
        return np.stack(state, axis=0)

    def other2array(self, state):
        if isinstance(state, dict):
            state = list(state.values())
        if not hasattr(self, 'keys') or self.keys is None:
            if isinstance(state, dict):
                self.keys = list(state.keys())
            else:
                self.keys = list(super().get_state().keys())
        if isinstance(state, list):
            state = np.stack(state, axis=0)
        return state

    def apply_rl_actions(self, rl_actions=None):
        if isinstance(rl_actions, np.ndarray):
            while rl_actions.ndim > 2:
                rl_actions.squeeze(axis=0)
            rl_actions = list(rl_actions)
        if isinstance(rl_actions, list):
            if self.keys is None:
                self.keys = list(super().get_state().keys())
            rl_actions = dict(zip(self.keys, rl_actions))
        super().apply_rl_actions(rl_actions)

    def step(self, rl_actions=None):
        if isinstance(rl_actions, np.ndarray):
            while rl_actions.ndim > 2:
                rl_actions.squeeze(axis=0)
            rl_actions = list(rl_actions)
        if isinstance(rl_actions, list):
            if self.keys is None:
                self.keys = list(super().get_state().keys())
            rl_actions = dict(zip(self.keys, rl_actions))
        s1, r, d, i = super().step(rl_actions)
        return self.other2array(s1), self.other2array(r), self.other2array(d), self.other2array(i)

    def init_neighbor_mask(self):
        delta = [
            [0, 0],
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1]
        ]
        for x in range(3):
            for y in range(3):
                for dx, dy in delta:
                    xx = x + dx
                    yy = y + dy
                    if 0 <= xx < 3 and 0 <= yy < 3:
                        self.neighbor_mask[y*3+x, yy*3+xx] = 1

    def init_distance_mask(self):
        block0 = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        block1 = block0 + 1
        block2 = block0 + 2
        row0 = np.hstack([block0, block1, block2])
        row1 = np.hstack([block1, block0, block1])
        row2 = np.hstack([block2, block1, block0])
        self.distance_mask = np.vstack([row0, row1, row2])

    def init(self):
        self.n_s_ls, self.n_a_ls, self.coop_gamma, self.distance_mask, self.neighbor_mask \
            = [], [], -1, np.zeros((9, 9)), np.zeros((9, 9))
        self.init_neighbor_mask()
        self.init_distance_mask()
        self.n_s_ls = [42] * 9
        self.n_a_ls = [2] * 9   

class FlowGridTestWrapper(FlowGridWrapper):

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            return rewards.desired_velocity(self)

def FlowGrid_Old(render=False):
    # Experiment parameters
    N_ROLLOUTS = 63  # number of rollouts per training iteration
    N_CPUS = 63  # number of parallel workers

    # Environment parameters
    HORIZON = 720  # time horizon of a single rollout
    V_ENTER = 30  # enter speed for departing vehicles
    INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
    LONG_LENGTH = 100  # length of final edge in route
    SHORT_LENGTH = 300  # length of edges that vehicles start on
    # number of vehicles originating in the left, right, top, and bottom edges
    N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 7, 10, 9, 8

    EDGE_INFLOW = 300  # inflow rate of vehicles at every edge
    N_ROWS = 3  # number of row of bidirectional lanes
    N_COLUMNS = 3  # number of columns of bidirectional lanes

    # we place a sufficient number of vehicles to ensure they confirm with the
    # total number specified above. We also use a "right_of_way" speed mode to
    # support traffic light compliance
    vehicles = VehicleParams()
    num_vehicles = (N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS
    vehicles.add(
        veh_id="human",
        acceleration_controller=(SimCarFollowingController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            max_speed=V_ENTER,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
        ),
        routing_controller=(GridRouter, {}),
        num_vehicles=num_vehicles)

    # inflows of vehicles are place on all outer edges (listed here)
    outer_edges = []
    outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
    outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
    outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
    outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

    # equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
    inflow = InFlows()
    for edge in outer_edges:
        inflow.add(
            veh_type="human",
            edge=edge,
            vehs_per_hour=EDGE_INFLOW,
            depart_lane="free",
            depart_speed=V_ENTER)

    flow_params = dict(
        # name of the experiment
        exp_tag="grid_0_{}x{}_i{}_multiagent".format(N_ROWS, N_COLUMNS, EDGE_INFLOW),

        # name of the flow environment the experiment is running on
        env_name=MultiTrafficLightGridPOEnv,

        # name of the network class the experiment is running on
        network=TrafficLightGridNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            restart_instance=True,
            sim_step=1,
            render=render,
            no_step_log=True,
            print_warnings=False
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            additional_params={
                "target_velocity": 50,
                "switch_time": 3,
                "num_observed": 2,
                "discrete": True,
                "tl_type": "actuated",
                "num_local_edges": 4,
                "num_local_lights": 4,
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            additional_params={
                "speed_limit": V_ENTER + 5,  # inherited from grid0 benchmark
                "grid_array": {
                    "short_length": SHORT_LENGTH,
                    "inner_length": INNER_LENGTH,
                    "long_length": LONG_LENGTH,
                    "row_num": N_ROWS,
                    "col_num": N_COLUMNS,
                    "cars_left": N_LEFT,
                    "cars_right": N_RIGHT,
                    "cars_top": N_TOP,
                    "cars_bot": N_BOTTOM,
                },
                "horizontal_lanes": 1,
                "vertical_lanes": 1,
            },
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization
        # or reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing='custom',
            shuffle=True,
        ),
    )
    create_env, env_name = make_create_env(params=flow_params, version=0)
    # Register as rllib env
    register_env(env_name, create_env)
    env = create_env()
    env.__class__ = FlowGridWrapper
    env.init()
    return env

def makeFlowGrid(version=0, render=None):
    # time horizon of a single rollout
    HORIZON = 400
    # inflow rate of vehicles at every edge
    EDGE_INFLOW = 300
    # enter speed for departing vehicles
    V_ENTER = 30
    # number of row of bidirectional lanes
    N_ROWS = 3
    # number of columns of bidirectional lanes
    N_COLUMNS = 3
    # length of inner edges in the grid network
    INNER_LENGTH = 300
    # length of final edge in route
    LONG_LENGTH = 100
    # length of edges that vehicles start on
    SHORT_LENGTH = 300
    # number of vehicles originating in the left, right, top, and bottom edges
    N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(SimCarFollowingController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            max_speed=V_ENTER,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
        ),
    routing_controller=(GridRouter, {}),
    num_vehicles=(N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS)
    
    outer_edges = []
    outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
    outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
    outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
    outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

    inflow = InFlows()
    for edge in outer_edges:
        inflow.add(
            veh_type="human",
            edge=edge,
            vehs_per_hour=EDGE_INFLOW,
            departLane="free",
            departSpeed=V_ENTER)

    flow_params = dict(
        # name of the experiment
        exp_tag="grid_0_{}x{}_i{}_multiagent".format(N_ROWS, N_COLUMNS, EDGE_INFLOW),

        # name of the flow environment the experiment is running on
        env_name=FlowGridWrapper,

        # name of the network class the experiment is running on
        network=TrafficLightGridNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            restart_instance=True,
            sim_step=1,
            render=False,
            no_step_log=True,
            print_warnings=False
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            additional_params={
                "target_velocity": 50,
                "switch_time": 3,
                "num_observed": 2,
                "discrete": True,
                "tl_type": "actuated",
                "num_local_edges": 4,
                "num_local_lights": 4
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            additional_params={
                "speed_limit": V_ENTER + 5,
                "grid_array": {
                    "short_length": SHORT_LENGTH,
                    "inner_length": INNER_LENGTH,
                    "long_length": LONG_LENGTH,
                    "row_num": N_ROWS,
                    "col_num": N_COLUMNS,
                    "cars_left": N_LEFT,
                    "cars_right": N_RIGHT,
                    "cars_top": N_TOP,
                    "cars_bot": N_BOTTOM,
                },
                "horizontal_lanes": 1,
                "vertical_lanes": 1,
            },
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing='custom',
            shuffle=True,
        ),
    )

    params = flow_params
    exp_tag = params["exp_tag"]
    base_env_name = params["env_name"].__name__

    # deal with multiple environments being created under the same name
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    while "{}-v{}".format(base_env_name, version) in env_ids:
        version += 1
    env_name = "{}-v{}".format(base_env_name, version)
    network_class = params["network"]

    env_params = params['env']
    net_params = params['net']
    initial_config = params.get('initial', InitialConfig())
    traffic_lights = params.get("tls", TrafficLightParams())
    

    def makeFlowGrid(*_):
        sim_params = deepcopy(params['sim'])
        vehicles = deepcopy(params['veh'])

        network = network_class(
            name=exp_tag,
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=traffic_lights,
        )

        # accept new render type if not set to None
        sim_params.render = render or sim_params.render

        entry_point = params["env_name"].__module__ + ':' + params["env_name"].__name__

        # register the environment with OpenAI gym
        register(
            id=env_name,
            entry_point=entry_point,
            kwargs={
                "env_params": env_params,
                "sim_params": sim_params,
                "network": network,
                "simulator": params['simulator']
            })

        return gym.envs.make(env_name)

    return makeFlowGrid, env_name

def makeFlowGridTest(version=0, render=None):
    # time horizon of a single rollout
    HORIZON = 400
    # inflow rate of vehicles at every edge
    EDGE_INFLOW = 300
    # enter speed for departing vehicles
    V_ENTER = 30
    # number of row of bidirectional lanes
    N_ROWS = 3
    # number of columns of bidirectional lanes
    N_COLUMNS = 3
    # length of inner edges in the grid network
    INNER_LENGTH = 300
    # length of final edge in route
    LONG_LENGTH = 100
    # length of edges that vehicles start on
    SHORT_LENGTH = 300
    # number of vehicles originating in the left, right, top, and bottom edges
    N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(SimCarFollowingController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            max_speed=V_ENTER,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
        ),
    routing_controller=(GridRouter, {}),
    num_vehicles=(N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS)
    
    outer_edges = []
    outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
    outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
    outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
    outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

    inflow = InFlows()
    for edge in outer_edges:
        inflow.add(
            veh_type="human",
            edge=edge,
            vehs_per_hour=EDGE_INFLOW,
            departLane="free",
            departSpeed=V_ENTER)

    flow_params = dict(
        # name of the experiment
        exp_tag="grid_0_{}x{}_i{}_multiagent".format(N_ROWS, N_COLUMNS, EDGE_INFLOW),

        # name of the flow environment the experiment is running on
        env_name=FlowGridTestWrapper,

        # name of the network class the experiment is running on
        network=TrafficLightGridNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            restart_instance=True,
            sim_step=1,
            render=False,
            no_step_log=True,
            print_warnings=False
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            additional_params={
                "target_velocity": 50,
                "switch_time": 3,
                "num_observed": 2,
                "discrete": True,
                "tl_type": "actuated",
                "num_local_edges": 4,
                "num_local_lights": 4
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            additional_params={
                "speed_limit": V_ENTER + 5,
                "grid_array": {
                    "short_length": SHORT_LENGTH,
                    "inner_length": INNER_LENGTH,
                    "long_length": LONG_LENGTH,
                    "row_num": N_ROWS,
                    "col_num": N_COLUMNS,
                    "cars_left": N_LEFT,
                    "cars_right": N_RIGHT,
                    "cars_top": N_TOP,
                    "cars_bot": N_BOTTOM,
                },
                "horizontal_lanes": 1,
                "vertical_lanes": 1,
            },
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing='custom',
            shuffle=True,
        ),
    )

    params = flow_params
    exp_tag = params["exp_tag"]
    base_env_name = params["env_name"].__name__

    # deal with multiple environments being created under the same name
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    while "{}-v{}".format(base_env_name, version) in env_ids:
        version += 1
    env_name = "{}-v{}".format(base_env_name, version)
    network_class = params["network"]

    env_params = params['env']
    net_params = params['net']
    initial_config = params.get('initial', InitialConfig())
    traffic_lights = params.get("tls", TrafficLightParams())
    

    def create_env(*_):
        sim_params = deepcopy(params['sim'])
        vehicles = deepcopy(params['veh'])

        network = network_class(
            name=exp_tag,
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=traffic_lights,
        )

        # accept new render type if not set to None
        sim_params.render = render or sim_params.render

        entry_point = params["env_name"].__module__ + ':' + params["env_name"].__name__

        # register the environment with OpenAI gym
        register(
            id=env_name,
            entry_point=entry_point,
            kwargs={
                "env_params": env_params,
                "sim_params": sim_params,
                "network": network,
                "simulator": params['simulator']
            })

        return gym.envs.make(env_name)

    return create_env, env_name
