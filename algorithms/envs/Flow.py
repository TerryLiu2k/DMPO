from ..envs.flow.envs.multiagent import MultiTrafficLightGridPOEnv, MultiAgentWaveAttenuationPOEnv
from ..envs.flow.networks import TrafficLightGridNetwork, RingNetwork
from ..envs.flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from ..envs.flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from ..envs.flow.controllers import SimCarFollowingController, GridRouter, RLController, IDMController, ContinuousRouter
from ray.tune.registry import register_env
from ..envs.flow.utils.registry import make_create_env

import numpy as np

class FlowGridWrapper(MultiTrafficLightGridPOEnv):
    def __get_state(self):
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

def FlowGrid():
    # Experiment parameters
    N_ROLLOUTS = 63  # number of rollouts per training iteration
    N_CPUS = 63  # number of parallel workers

    # Environment parameters
    HORIZON = 400  # time horizon of a single rollout
    V_ENTER = 30  # enter speed for departing vehicles
    INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
    LONG_LENGTH = 100  # length of final edge in route
    SHORT_LENGTH = 300  # length of edges that vehicles start on
    # number of vehicles originating in the left, right, top, and bottom edges
    N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1

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
            render=False,
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
    return env


class RingAttenuationWrapper(MultiAgentWaveAttenuationPOEnv):
    def state(self):
        return self.get_state()


def RingAttenuation():
    # time horizon of a single rollout
    HORIZON = 3000
    # number of rollouts per training iteration
    N_ROLLOUTS = 20
    # number of parallel workers
    N_CPUS = 2
    # number of automated vehicles. Must be less than or equal to 22.
    NUM_AUTOMATED = 2

    # We evenly distribute the automated vehicles in the network.
    num_human = 22 - NUM_AUTOMATED
    humans_remaining = num_human

    vehicles = VehicleParams()
    for i in range(NUM_AUTOMATED):
        # Add one automated vehicle.
        vehicles.add(
            veh_id="rl_{}".format(i),
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=1)

        # Add a fraction of the remaining human vehicles.
        vehicles_to_add = round(humans_remaining / (NUM_AUTOMATED - i))
        humans_remaining -= vehicles_to_add
        vehicles.add(
            veh_id="human_{}".format(i),
            acceleration_controller=(IDMController, {
                "noise": 0.2
            }),
            car_following_params=SumoCarFollowingParams(
                min_gap=0
            ),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=vehicles_to_add)

    flow_params = dict(
        # name of the experiment
        exp_tag="multiagent_ring",

        # name of the flow environment the experiment is running on
        env_name=MultiAgentWaveAttenuationPOEnv,

        # name of the network class the experiment is running on
        network=RingNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.1,
            render=False,
            restart_instance=False
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            warmup_steps=750,
            clip_actions=False,
            additional_params={
                "max_accel": 1,
                "max_decel": 1,
                "ring_length": [220, 270],
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params={
                "length": 260,
                "lanes": 1,
                "speed_limit": 30,
                "resolution": 40,
            }, ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(),
    )

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)
    return RingAttenuationWrapper(create_env())
