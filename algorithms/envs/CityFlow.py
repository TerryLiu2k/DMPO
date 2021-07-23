import os
import copy
import numpy as np
from gym.spaces import Discrete, Box
from ..envs.cityflow.anon_env import AnonEnv


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def concat(d:dict):
    l = []
    for v in d.values():
        l += v
    return l

class CFGrid6_6(AnonEnv):
    def __init__(self):
        memo = '0515_afternoon_Colight_6_6_bi'
        gui = False
        road_net = '6_6'
        volume = 300
        suffix = '0.3_bi'
        dic_traffic_env_conf = {
            "ACTION_PATTERN": "set",
            "NUM_INTERSECTIONS": 1,
            "MIN_ACTION_TIME": 10,
            "YELLOW_TIME": 5,
            "ALL_RED_TIME": 0,
            "NUM_PHASES": 2,
            "NUM_LANES": 1,
            "ACTION_DIM": 2,
            "MEASURE_TIME": 10,
            "IF_GUI": True,
            "DEBUG": False,

            "INTERVAL": 1,
            "THREADNUM": 8,
            "SAVEREPLAY": True,
            "RLTRAFFICLIGHT": True,

            "DIC_FEATURE_DIM": dict(
                D_LANE_QUEUE_LENGTH=(4,),
                D_LANE_NUM_VEHICLE=(4,),

                D_COMING_VEHICLE=(4,),
                D_LEAVING_VEHICLE=(4,),

                D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                D_CUR_PHASE=(1,),
                D_NEXT_PHASE=(1,),
                D_TIME_THIS_PHASE=(1,),
                D_TERMINAL=(1,),
                D_LANE_SUM_WAITING_TIME=(4,),
                D_VEHICLE_POSITION_IMG=(4, 60,),
                D_VEHICLE_SPEED_IMG=(4, 60,),
                D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                D_PRESSURE=(1,),

                D_ADJACENCY_MATRIX=(2,)
            ),

            "LIST_STATE_FEATURE": [
                "cur_phase",
                "time_this_phase",
                "vehicle_position_img",
                "vehicle_speed_img",
                "vehicle_acceleration_img",
                "vehicle_waiting_time_img",
                "lane_num_vehicle",
                "lane_num_vehicle_been_stopped_thres01",
                "lane_num_vehicle_been_stopped_thres1",
                "lane_queue_length",
                "lane_num_vehicle_left",
                "lane_sum_duration_vehicle_left",
                "lane_sum_waiting_time",
                "terminal",

                "coming_vehicle",
                "leaving_vehicle",
                "pressure",

                "adjacency_matrix",
                "adjacency_matrix_lane"

            ],

            "DIC_REWARD_INFO": {
                "flickering": 0,
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0,
            },

            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": [
                'WSES',
                'NSSS',
                'WLEL',
                'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'NSNL',
                # 'SSSL',
            ],

        }
        hangzhou_archive = False
        cnt = 3600
        gen = 4
        r_all = False
        workers = 7
        onemodel = False
        TOP_K_ADJACENCY = 5
        TOP_K_ADJACENCY_LANE=5
        NUM_ROUNDS=100
        EARLY_STOP=False
        NEIGHBOR=False
        SAVEREPLAY=False
        ADJACENCY_BY_CONNECTION_OR_GEO=False
        PRETRAIN=False
        _LS = {"LEFT": 1,
               "RIGHT": 0,
               "STRAIGHT": 1
               }
        _S = {
            "LEFT": 0,
            "RIGHT": 0,
            "STRAIGHT": 1
        }
        _LSR = {
            "LEFT": 1,
            "RIGHT": 1,
            "STRAIGHT": 1
        }

        NUM_COL = int(road_net.split('_')[0])
        NUM_ROW = int(road_net.split('_')[1])
        num_intersections = NUM_ROW * NUM_COL
        print('num_intersections:', num_intersections)
        ENVIRONMENT = "anon"

        traffic_file_list = ["{0}_{1}_{2}_{3}".format(ENVIRONMENT, road_net, volume, suffix)]
        traffic_file_list = [i + ".json" for i in traffic_file_list]

        process_list = []
        n_workers = workers     #len(traffic_file_list)
        multi_process = True
        ANON_PHASE_REPRE={
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0]
        }

        for traffic_file in traffic_file_list:
            dic_traffic_env_conf_extra = {
                "USE_LANE_ADJACENCY": True,
                "ONE_MODEL": onemodel,
                "NUM_AGENTS": num_intersections,
                "NUM_INTERSECTIONS": num_intersections,
                "ACTION_PATTERN": "set",
                "MEASURE_TIME": 10,
                "IF_GUI": gui,
                "DEBUG": False,
                "TOP_K_ADJACENCY": TOP_K_ADJACENCY,
                "ADJACENCY_BY_CONNECTION_OR_GEO": ADJACENCY_BY_CONNECTION_OR_GEO,
                "TOP_K_ADJACENCY_LANE": TOP_K_ADJACENCY_LANE,
                "SIMULATOR_TYPE": ENVIRONMENT,
                "BINARY_PHASE_EXPANSION": True,
                "FAST_COMPUTE": True,

                "NEIGHBOR": NEIGHBOR,
                "MODEL_NAME": None,



                "SAVEREPLAY": SAVEREPLAY,
                "NUM_ROW": NUM_ROW,
                "NUM_COL": NUM_COL,

                "TRAFFIC_FILE": traffic_file,
                "VOLUME": volume,
                "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

                "phase_expansion": {
                    1: [0, 1, 0, 1, 0, 0, 0, 0],
                    2: [0, 0, 0, 0, 0, 1, 0, 1],
                    3: [1, 0, 1, 0, 0, 0, 0, 0],
                    4: [0, 0, 0, 0, 1, 0, 1, 0],
                    5: [1, 1, 0, 0, 0, 0, 0, 0],
                    6: [0, 0, 1, 1, 0, 0, 0, 0],
                    7: [0, 0, 0, 0, 0, 0, 1, 1],
                    8: [0, 0, 0, 0, 1, 1, 0, 0]
                },

                "phase_expansion_4_lane": {
                    1: [1, 1, 0, 0],
                    2: [0, 0, 1, 1],
                },


                "LIST_STATE_FEATURE": [
                    "cur_phase",
                    # "time_this_phase",
                    # "vehicle_position_img",
                    # "vehicle_speed_img",
                    # "vehicle_acceleration_img",
                    # "vehicle_waiting_time_img",
                    "lane_num_vehicle",
                    # "lane_num_vehicle_been_stopped_thres01",
                    # "lane_num_vehicle_been_stopped_thres1",
                    # "lane_queue_length",
                    # "lane_num_vehicle_left",
                    # "lane_sum_duration_vehicle_left",
                    # "lane_sum_waiting_time",
                    # "terminal",
                    # "coming_vehicle",
                    # "leaving_vehicle",
                    # "pressure"

                    # "adjacency_matrix",
                    # "lane_queue_length",
                    # "connectivity",

                    # adjacency_matrix_lane
                ],

                    "DIC_FEATURE_DIM": dict(
                        D_LANE_QUEUE_LENGTH=(4,),
                        D_LANE_NUM_VEHICLE=(4,),

                        D_COMING_VEHICLE = (12,),
                        D_LEAVING_VEHICLE = (12,),

                        D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                        D_CUR_PHASE=(1,),
                        D_NEXT_PHASE=(1,),
                        D_TIME_THIS_PHASE=(1,),
                        D_TERMINAL=(1,),
                        D_LANE_SUM_WAITING_TIME=(4,),
                        D_VEHICLE_POSITION_IMG=(4, 60,),
                        D_VEHICLE_SPEED_IMG=(4, 60,),
                        D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                        D_PRESSURE=(1,),

                        D_ADJACENCY_MATRIX=(2,),

                        D_ADJACENCY_MATRIX_LANE=(6,),

                    ),

                "DIC_REWARD_INFO": {
                    "flickering": 0,#-5,#
                    "sum_lane_queue_length": 0,
                    "sum_lane_wait_time": 0,
                    "sum_lane_num_vehicle_left": 0,#-1,#
                    "sum_duration_vehicle_left": 0,
                    "sum_num_vehicle_been_stopped_thres01": 0,
                    "sum_num_vehicle_been_stopped_thres1": -0.25,
                    "pressure": 0  # -0.25
                },

                "LANE_NUM": {
                    "LEFT": 1,
                    "RIGHT": 1,
                    "STRAIGHT": 1
                },

                "PHASE": {
                    "sumo": {
                        0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                        1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                        2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                        3: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                    },

                    # "anon": {
                    #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                    #     1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],# 'WSES',
                    #     2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],# 'NSSS',
                    #     3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],# 'WLEL',
                    #     4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]# 'NLSL',
                    #     # 'WSWL',
                    #     # 'ESEL',
                    #     # 'WSES',
                    #     # 'NSSS',
                    #     # 'NSNL',
                    #     # 'SSSL',
                    # },
                    "anon":ANON_PHASE_REPRE,
                    # "anon": {
                    #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                    #     1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    #     2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    #     3: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                    #     4: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                    #     # 'WSWL',
                    #     # 'ESEL',
                    #     # 'WSES',
                    #     # 'NSSS',
                    #     # 'NSNL',
                    #     # 'SSSL',
                    # },
                }
            }
            template = "template_lsr"
            dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

            if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,)
                if dic_traffic_env_conf_extra['NEIGHBOR']:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (8,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (8,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (8,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (8,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)
                else:

                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)

            deploy_dic_traffic_env_conf = merge(dic_traffic_env_conf, dic_traffic_env_conf_extra)
        super().__init__(
            path_to_log=None,
            path_to_work_directory=None, # self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=deploy_dic_traffic_env_conf
        )
        self.action_space = Discrete(2)
        self.reset()
        self.observation_space = Box(low=-0.0, high=np.inf, shape=[len(self.get_state()[0])], dtype=np.float32)

    def get_state(self):
        state = super(CFGrid6_6, self).get_state()
        state = [np.array(concat(s)) for s in state]
        return np.array(state)
