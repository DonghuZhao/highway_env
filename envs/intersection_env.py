import random
from typing import Dict, Tuple

from gymnasium.envs.registration import register
import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle

from highway_env.envs.common.observation_utils import ObservationUtils

class IntersectionEnv(AbstractEnv):

    ACTIONS: Dict[int, str] = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "lat_off", "ang_off"],
                "features": ["x", "y", "speed", "heading","lat_off", "ang_off"],        # state_v3
                # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "lat_off", "ang_off", "EHMI", "Safety"],
                "features_range": {
                    "x": [-20, 60],
                    "y": [-50, 30],
                    "vx": [-10, 10],
                    "vy": [-10, 10],
                    "speed": [0, 10],
                    "heading": [-np.pi, np.pi],
                    "cos_h": [-1, 1],
                    "sin_h": [-1, 1],
                    "lat_off": [-2, 3],
                    "ang_off": [-np.pi / 4, np.pi / 4],
                    "Safety":  [0, 150],
                    "distance": [0, 30],
                    "angle": [0, 360],
                    "ob_speed": [0, 10],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False,
                "normalize": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                # "target_speeds": [0, 4.5, 9],
                "steering_range": [-np.pi / 4, np.pi / 4],
                "dynamical": True,
                "EHMI": True
            },
            "duration": 20,  # [s]
            "type": "straight",
            "destination": "o2",        # left:"o2", straight:"
            "simulation_frequency": 10,  # [Hz] # 15
            "policy_frequency": 5,  # [Hz]  # 1
            "controlled_vehicles": 1,
            "initial_vehicle_count": 20,
            "spawn_probability": 0.6,
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.7, 0.6],       # 0.2, 0.6
            "scaling": 7.5 * 1.8,
            "collision_reward": -10,  # -3
            "high_speed_reward": 3, # 3  #8
            "arrived_reward": 10,
            "reward_speed_range": [3.0, 10.0],  # v2: 3.0
            "pet_reward": 5,    #5
            "pass_time_reward": 5,  #5
            "normalize_reward": True,   # False
            "offroad_terminal": True,
            "real_time_rendering": True,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle" # LinearVehicle
        })
        return config

    def __init__(self, *args, **kwargs):
        # 添加子类的新属性
        self.other_vehicles = []
        self.EHMI = None
        self.EHMI_CHANGED = False
        self.EHMISHOW_STEPS = 0
        self.EHMI_REWARD = False
        self.ego_pass_time = None
        self.other_pass_time = None
        self.ego_travel_time = None
        self.pass_stop_line = False
        # 调用父类的 __init__ 方法，传递所有参数
        super().__init__(*args, **kwargs)

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        # 将车辆速度映射到指定的速度范围内，得到一个标准化的速度值 scaled_speed
        scaled_speed = utils.lmap(self.vehicle.speed, self.get_wrapper_attr('config')["reward_speed_range"], [0, 1])
        ang_off = abs(vehicle.lane_offset[2])
        lat_off = abs(vehicle.lane_offset[1])

        #进口道总长度
        d1 = 20 #m
        #左转车道长度
        d2 = 32.4 #m
        #北出口道长度
        d3 = 12.8 #m
        #车辆在车道行驶距离
        d_ego = vehicle.lane.local_coordinates(vehicle.position)[0]
        # print(vehicle.lane_index)
        disfo = 0
        if vehicle.lane_index == ('o1', 'ir1', 0) :
            disfo = d_ego
        if vehicle.lane_index == ('ir1', 'il2', 0):
            disfo = d_ego + d1
        if vehicle.lane_index == ('il2', 'o2', 0):
            disfo = d_ego + d2 + d1
        navi = disfo/(d1+d2+d3) #一个0到1的数

        # 10 * ang_off ** 2
        # reward_v1
        reward = self.get_wrapper_attr('config')["collision_reward"] * vehicle.crashed \
                 + self.get_wrapper_attr('config')["high_speed_reward"] * np.clip(scaled_speed, 0, 1)\
                 + 1 / (1 + 10 * ang_off ** 2) \
                 + navi

        # # reward_v6
        # reward = self.get_wrapper_attr('config')["collision_reward"] * vehicle.crashed \
        #          + self.get_wrapper_attr('config')["high_speed_reward"] * np.clip(scaled_speed, 0, 1)\
        #          + 1 / (1 + 10 * ang_off ** 2)

        # # reward_v2
        # # 车速过低时的惩罚
        # if self.vehicle.speed < 2:
        #     reward -= navi
        # reward_v3
        # if navi == 0:
        #     reward -= 3
        # reward_v4
        if not self.ego_pass_time or not self.other_pass_time:
            reward_ = self._pet_reward()
            # if reward_ != 0:
            #     print("pet_reward:", reward_)
            reward += reward_
        # reward_v5
        if not self.ego_travel_time:
            reward_ = self._passtime_reward()
            # if reward_ != 0:
            #     print("passtime_reward:", reward_)
            reward += reward_

        # reward_v7
        if vehicle.crashed:
            # print("crash")
            reward = self.get_wrapper_attr('config')["collision_reward"]

        # # reward_v8
        # pet = self._safety_sensor()
        # # print("pet_reward:", pet)
        # reward += pet

        # reward_v9
        reward += self._stop_line_reward(ego_type=self.get_wrapper_attr('config')["type"])
        # reward += self._stop_line_reward() * 2 #EHMI defensive

        # reward_v10
        reward_ = self._EHMI_reward()
        if reward_ != 0:
            self.EHMI_REWARD = True
        reward += reward_

        # reward_v11
        if self.EHMI_CHANGED:
            reward -= 3


        # 如果车辆已经到达目的地，则将到达奖励 self.get_wrapper_attr('config')["arrived_reward"] 赋值给总奖励
        # print(vehicle.lane_index, vehicle.lane.local_coordinates(vehicle.position)[0])
        if self.has_arrived(vehicle):
            reward = self.get_wrapper_attr('config')["arrived_reward"]
            # print("arrived")
        if self.get_wrapper_attr('config')["normalize_reward"]:
            reward = utils.lmap(reward, [self.get_wrapper_attr('config')["collision_reward"], self.get_wrapper_attr('config')["arrived_reward"]], [-1, 1])
        reward = 0 if not vehicle.on_road else reward

        reward = 0 if self._is_truncated() and not self.has_arrived(vehicle) else reward

        reward = 0 if self._is_terminated() and not self.has_arrived(vehicle) else reward
        # print("norm_reward:", reward)

        return reward

    def _pet_reward(self):
        """给出冲突的稀疏奖励"""
        if self.get_wrapper_attr('config')['type'] == "left":
            # 冲突点 左转车 y:-17.5，
            # 直行车 x: 3
            if self.vehicle.position[1] < -17.5:
                if not self.ego_pass_time:
                    self.ego_pass_time = self.steps
            if len(self.road.vehicles) == 2:
                if self.road.vehicles[0].position[0] < 3:
                    if not self.other_pass_time:
                        self.other_pass_time = self.steps
            if self.ego_pass_time and self.other_pass_time:
                # print('pet:', abs(self.other_pass_time - self.ego_pass_time))
                return self.get_wrapper_attr('config')["pet_reward"] * abs(self.other_pass_time - self.ego_pass_time) / 100
        elif self.get_wrapper_attr('config')['type'] == "straight":
            if self.vehicle.position[0] < 3:
                if not self.ego_pass_time:
                    self.ego_pass_time = self.steps
            if len(self.road.vehicles) == 2:
                if self.road.vehicles[0].position[1] < -17.5:
                    if not self.other_pass_time:
                       self.other_pass_time = self.steps
            if self.ego_pass_time and self.other_pass_time:
                # print('pet:', abs(self.other_pass_time - self.ego_pass_time))
                pet = self.other_pass_time - self.ego_pass_time
                # return self.get_wrapper_attr('config')["pet_reward"] * abs(pet) / 100
                return self.get_wrapper_attr('config')["pet_reward"] * max(min(pet, 50), -50) / 50
        # print(self.ego_pass_time, self.other_pass_time)
        return 0

    def _passtime_reward(self):
        """给出效率的稀疏奖励"""
        target_lane = ('il2', 'o2', 0)
        MAXPASSTIME = 200
        pass_reward = self.get_wrapper_attr('config')["pass_time_reward"]
        if self.get_wrapper_attr('config')['type'] == "straight":
            target_lane = ('il1', 'o1', 0)
            pass_reward = 10

        if self.vehicle.lane_index == target_lane:
            self.ego_travel_time = self.steps
        if self.ego_travel_time:
            # print('travel_time:', self.ego_travel_time)
            return pass_reward * (1 - self.ego_travel_time / MAXPASSTIME)
        return 0

    def _safety_sensor(self) -> float:
        """
        Compute the safety sensor of a vehicle.
        """
        x, y = self.vehicle.position
        obx, oby = self.road.vehicles[0].position
        v = self.vehicle.speed
        yaw = self.vehicle.heading
        obv = self.road.vehicles[0].speed
        obyaw = self.road.vehicles[0].heading

        distance = ObservationUtils.calculate_distance(x, y, obx, oby)
        if distance > 60:
            return 0
        pet = ObservationUtils.calculate_pre_pet(x, y, v, yaw, obx, oby, obv, obyaw, ego_type="left")
        # print("pet:", pet)
        pet = abs(pet)
        if pet > 10:
            return 0
        # if distance > 30:
        #     return - (10 - pet) * 0.3 * (60 - distance) / 60
        return - (10 - pet) * 0.3 * (60 - distance) / 60

    def _stop_line_reward(self, ego_type="left") -> float:

        x, y = self.vehicle.position
        obx, oby = self.road.vehicles[0].position
        v = self.vehicle.speed
        yaw = self.vehicle.heading
        obv = self.road.vehicles[0].speed
        obyaw = self.road.vehicles[0].heading

        if self.pass_stop_line:
            return 0
        if (ego_type == "left" and x < -1) or (ego_type == "straight" and x > 31):
            return 0
        self.pass_stop_line = True
        reward = self.get_wrapper_attr('config')['arrived_reward'] / 2
        pet = ObservationUtils.calculate_pre_pet(x, y, v, yaw, obx, oby, obv, obyaw, ego_type=ego_type)
        # print("pet:", pet)
        pet = abs(pet)
        if pet > 5:
            return reward
        return reward - (5 - pet) * 0.5

    def _EHMI_reward(self) -> float:
        if self.EHMI_REWARD:
            return 0
        if self.ego_pass_time and not self.other_pass_time:
            if self.EHMI == 'Y':
                return -5
            elif self.EHMI == 'R':
                return 5
        elif not self.ego_pass_time and self.other_pass_time:
            if self.EHMI == 'R':
                return -5
            elif self.EHMI == 'Y':
                return 5
        return 0

    def _is_terminated(self) -> bool:
        # if any(vehicle.crashed for vehicle in self.controlled_vehicles):
        #     print("crash")
        # if all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles):
        #     print("arrived")
        # if self.steps >= self.get_wrapper_attr('config')["duration"] * self.get_wrapper_attr('config')["policy_frequency"]:
        #     print("duration")
        # if (self.get_wrapper_attr('config')["offroad_terminal"] and not self.vehicle.on_road):
        #     print("offroad")
        # if (self.vehicle.lane_index != ('o1', 'ir1', 0)) and (self.vehicle.lane_index != ('ir1', 'il2', 0)) and (
        #         self.vehicle.lane_index != ('il2', 'o2', 0)):
        #     print("offlane")
        # if self.vehicle.speed < 0:
        #     print("speed<0")
        reverse_speed = False
        speeds = [v.speed for v in self.road.vehicles]
        if min(speeds) < 0:
            reverse_speed = True
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
               or self.steps >= self.get_wrapper_attr('config')["duration"] * self.get_wrapper_attr('config')["policy_frequency"] \
               or (self.get_wrapper_attr('config')["offroad_terminal"] and not self.vehicle.on_road)\
               or self._agent_is_out_of_boundary() \
               or self.vehicle.speed < 0 \
               or reverse_speed

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
            or self.steps >= self.get_wrapper_attr('config')["duration"] * self.get_wrapper_attr('config')["policy_frequency"] \
            or self.has_arrived(vehicle)

    #车道边界终止条件
    def _agent_is_out_of_boundary(self) -> bool:
        """The episode is over when ego vehicle cross the boundary of the road."""
        if self.get_wrapper_attr('config')["type"] == "straight":
            return (self.vehicle.lane_index != ('o3', 'ir3', 0)) and (
                        self.vehicle.lane_index != ('ir3', 'il1', 0)) and (
                    self.vehicle.lane_index != ('il1', 'o1', 0))

        return (self.vehicle.lane_index != ('o1', 'ir1', 0)) and (self.vehicle.lane_index != ('ir1', 'il2', 0)) and (
                self.vehicle.lane_index != ('il2', 'o2', 0))

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.get_wrapper_attr('config')["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        return info

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.get_wrapper_attr('config')["initial_vehicle_count"])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:

        obs, reward, done, truncated, info = super().step(action)
        self._clear_vehicles()

        return obs, reward, done, truncated, info

    def update_EHMI(self, EHMI=None):
        """
        根据策略网络的EHMI输出更新环境中的EHMI
        distance > 80 时不显示EHMI
        EHMI显示的最短时间为2s（2s内不允许更换内容)
        """
        self.EHMI_CHANGED = False # 初始化
        distance = ObservationUtils.calculate_distance(self.vehicle.position[0],
                                                       self.vehicle.position[1],
                                                       self.road.vehicles[0].position[0],
                                                       self.road.vehicles[0].position[1])
        ego_distance2conflict = ObservationUtils.calculate_distance(self.vehicle.position[0],
                                                                   self.vehicle.position[1],
                                                                   3, -17.5)
        if (distance > 80
                or self.ego_pass_time is not None
                or self.other_pass_time is not None):   # 不显示EHMI
            self.EHMI = None
            self.EHMISHOW_STEPS = 0
        elif self.EHMI is None:
            self.EHMI = EHMI
            self.EHMISHOW_STEPS += 1
        elif (self.EHMISHOW_STEPS < 2 * self.unwrapped.config["policy_frequency"]
              or self.EHMI == EHMI or ego_distance2conflict < 10):
            self.EHMISHOW_STEPS += 1
        else:
            self.EHMISHOW_STEPS = 0
            self.EHMI = EHMI
            self.EHMI_CHANGED = True

        # EHMI信息传递给其他背景车
        for v in self.road.vehicles:
            v.EHMI = self.EHMI

    def _make_road(self) -> None:

        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
# 西方向
    # 进口道
        net.add_lane("o1", "ir1", StraightLane([-41.63, -14.3], [-1.63, -14.3], line_types=[c, s],
                                               width=3, priority=0, speed_limit=30))
        net.add_lane("o1a", "ir1a", StraightLane([-41.63, -11.3], [-1.63, -11.3], width=3,
                                                line_types=[n, c], priority=0, speed_limit=30))
    # 连接段
    #     # 2条直行车道
    #     net.add_lane("ir1", "il3",
    #                  StraightLane([-1.63, -14.3], [31.14, -14.9], width=3, line_types=[n, n], priority=0,
    #                               speed_limit=30))
        net.add_lane("ir1a", "il3a",
                     StraightLane([-1.63, -11.3], [31.14, -11.9], width=3, line_types=[n, n], priority=0,
                                  speed_limit=30))
        # 右转车道,连接南出口道
        net.add_lane("ir1a", "il0",
                     CircularLane([-1.3, 2.7], 14.1, np.radians(265), np.radians(350), width=3,
                                  line_types=[n, c], priority=0, speed_limit=30))
        # 左转车道,连接北出口道
        net.add_lane("ir1", "il2",
                     CircularLane([-2, -31], 17, np.radians(95), np.radians(-5), width=5,clockwise=False,
                                  line_types=[n, n], priority=0, speed_limit=15))
    # 右侧出口道
        net.add_lane("il0", "o0",
                     StraightLane([12.95, 0.33], [12.95, 10.5], width=3.5,line_types=[n, c], priority=0,
                                  speed_limit=30))
# 东方向
    # 进口道
        net.add_lane("o3", "ir3", StraightLane([81.14, -17.9], [31.14, -17.9], line_types=[n, n],
                                               width=3, priority=0, speed_limit=30))
        net.add_lane("o3a", "ir3a", StraightLane([81.14, -20.9], [31.14, -20.9], width=3,
                                                 line_types=[s, c], priority=0, speed_limit=30))
    # 连接段
        # 2条直行车道
        net.add_lane("ir3", "il1",
                     StraightLane([31.14, -17.9], [-1.63, -17.3], width=3, line_types=[n, n], priority=0,
                                  speed_limit=30))
        net.add_lane("ir3a", "il1a",
                     StraightLane([31.14, -20.9], [-1.63, -20.3], width=3, line_types=[n, n], priority=0,
                                  speed_limit=30))
        # 右转车道,连接北出口道
        net.add_lane("ir3a", "il2",
                     CircularLane([29.65, -35.5], 15, np.radians(85), np.radians(170), width=3.7,
                                  line_types=[n, c], priority=0, speed_limit=30))
    # 右侧出口道
        net.add_lane("il2", "o2",
                     StraightLane([14.81, -32.28], [14.81, -42.45], width=4, line_types=[n, c], priority=0,
                                  speed_limit=30))
# 南方向
     # 进口道
        net.add_lane("o0", "ir0", StraightLane([16.7, 10.33], [16.7, 0.1], line_types=[c, c],
                                               width=4, priority=0, speed_limit=30))
     # 连接段
        net.add_lane("ir0", "il2",
                     StraightLane([16.7, 0.33], [14.81, -32.28], width=4, line_types=[n, n], priority=0,
                                  speed_limit=30))
        # 右转车道,连接南出口道
        net.add_lane("ir0", "il3a",
                     CircularLane([29.9, 0.8], 13, np.radians(180), np.radians(275), width=3.7,
                                  line_types=[n, c], priority=0, speed_limit=30))
        # 左转车道,连接西出口道
        net.add_lane("ir0", "il1",
                     CircularLane([-2, 0], 18.5, np.radians(360), np.radians(270), width=5,clockwise=False,
                                  line_types=[n, n], priority=0, speed_limit=30))
     # 右侧出口道
        net.add_lane("il3", "o3",
                     StraightLane([31.14, -14.9], [81.14, -14.9], width=3, line_types=[c, s], priority=0,
                                  speed_limit=30))
        net.add_lane("il3a", "o3",
                     StraightLane([31.14, -11.9], [81.14, -11.9], width=3, line_types=[n, c], priority=0,
                                  speed_limit=30))
# 北方向
    # 进口道
        net.add_lane("o2", "ir2", StraightLane([11, -42.28], [11, -32.1], line_types=[n, c],
                                               width=3.6, priority=0, speed_limit=30))
    # 连接段
        net.add_lane("ir2", "il0",
                     StraightLane([11, -32.28], [12.95, 0.33], width=3.5, line_types=[n, n], priority=0,
                                  speed_limit=30))
        # 右转车道,连接西出口道
        net.add_lane("ir2", "il1a",
                     CircularLane([-0.55, -31.55], 11.671, np.radians(-5), np.radians(95), width=3.5,
                                  line_types=[n, c], priority=0, speed_limit=30))
    # 右侧出口道
        net.add_lane("il1", "o1",
                     StraightLane([-1.63, -17.2], [-41.63, -17.3], width=3, line_types=[n, n], priority=0,
                                  speed_limit=30))
        net.add_lane("il1a", "o1",
                     StraightLane([-1.63, -20.3], [-41.63, -20.3], width=3, line_types=[s, c], priority=0,
                                  speed_limit=30))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.get_wrapper_attr('config')["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:

        # Configure vehicles
        vehicle_type = utils.class_from_path(self.get_wrapper_attr('config')["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -6

        #选择主车进入的交叉口
        # mode = 's2n'
        mode = 'w2e'
        mode = 'e2w' if self.get_wrapper_attr('config')["type"] == 'straight' else 'w2e'
        if mode == 'w2e':
            ego_lane = ("o1", "ir1", 0)
            ego_target = "o2"
            logitudinal, speed = self._vehicle_generator()
            #东侧直行车
            self._spawn_vehicle(logitudinal, speed, spawn_probability=1, go_straight=True, position_deviation=0,
                                speed_deviation=0, lane_index=("o3", "ir3", 0)) #longitudinal = 8 # speed = 10  # position_Deviation = 0.1 # 单机（0， 0）
            # #西侧直行车
            # self._spawn_vehicle(5, lane_index=("ir1a", "il3a", 0), dest_index="o3", speed=12)#纵向最小为5
            # # 由东侧进口道直行
            # self._spawn_vehicle(25, lane_index=("ir3a", "il1a", 0), dest_index="o1", speed=8)
            # # 西出口道直行
            # self._spawn_vehicle(3, lane_index=("il1", "o1", 0), dest_index="o1", speed=8)
            # #西出口道右转
            # self._spawn_vehicle(7, lane_index=("o1a", "ir1a", 0), dest_index="o0", speed=0)
            # self._spawn_vehicle(11, lane_index=("ir1a", "il0", 0), dest_index="o0", speed=9)
            #
            # for t in range(3):
            #     #由东侧进口道右转到北
            #     self._spawn_vehicle([1, 5, 14][t], lane_index=("o3a", "ir3a", 0), dest_index="o2", speed=15)
        elif mode == "e2w":
            ego_lane = ("o3", "ir3", 0)
            ego_target = "o1"
            logitudinal, speed = self._vehicle_generator()
            # 西侧左转车     # 20
            self._spawn_vehicle(logitudinal, speed, spawn_probability=1, dest_index="o2", position_deviation=0.1,
                                speed_deviation=0, lane_index=("o1", "ir1", 0))

        else:
            ego_lane = ("o0", "ir0", 0)
            ego_target = "o1"
            #北侧直行车
            self._spawn_vehicle(8, 8, spawn_probability=1, dest_index="o0", position_deviation=0.1,
                                speed_deviation=0, lane_index=("o2", "ir2", 0))
            #西出口道右转
            self._spawn_vehicle(7, lane_index=("o1a", "ir1a", 0), dest_index="o0", speed=12)
            self._spawn_vehicle(11, lane_index=("ir1a", "il0", 0), dest_index="o0", speed=9)
            for t in range(3):
                #由东侧进口道右转到北
                self._spawn_vehicle([1, 5, 14][t], lane_index=("o3a", "ir3a", 0), dest_index="o2", speed=15)

        # Controlled vehicles
        self.controlled_vehicles = []
        logitudinal = 20 if self.get_wrapper_attr('config')['type'] == 'straight' else 5
        for ego_id in range(0, self.get_wrapper_attr('config')["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(ego_lane)
            ego_vehicle = self.action_type.vehicle_class(
                             self.road,
                             ego_lane.position(logitudinal, 0),  # -1        #5logi
                             speed=6, # 0
                             heading=ego_lane.heading_at(60))
            try:
                ego_vehicle.plan_route_to(ego_target)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 3:
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       speed: float = 5,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       lane_index: str = ("o3a", "ir3a", 0),
                       dest_index: str = "o1",
                       go_straight: bool = False) -> None:

        vehicle_type = utils.class_from_path(self.get_wrapper_attr('config')["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, lane_index,
                                            longitudinal=longitudinal,
                                            speed=speed)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 5:
                return
        vehicle.plan_route_to(dest_index)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _vehicle_generator(self):

        discrete = False
        if discrete:
            # v1
            logitudinal = [0, 15, 30]
            speed = [4, 7, 10]
            # 匀速运行结果
            # logitudinal：   0     15      30
            # speed：4        L      L       ×
            # speed： 7       L      ×       S
            # speed： 10      ×      ×       S
            i = random.choice(range(len(logitudinal)))
            i = 0
            j = random.choice(range(len(speed)))
            j = 1
            return logitudinal[i], speed[j]
        else:
            logitudinal = np.random.randint(10, 30) #10-30
            speed = np.random.randint(6, 8)
            if self.get_wrapper_attr('config')["type"] == "straight":
                logitudinal = np.random.randint(0, 25)
                speed = np.random.randint(4, 8) # 6 - 8
            # logitudinal = 25
            # speed = 8
            return logitudinal, speed

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 0.5 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 10) -> bool:     # default: 25
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)


class ContinuousIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config['action'].update({
                    "type": "ContinuousAction",
                    "steering_range": [-np.pi / 4, np.pi / 4], # 3
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True
                })
        # config.update({
        #     "observation": {
        #         "type": "Kinematics",
        #         # "vehicles_count": 5,
        #         # "features": ["presence", "x", "y", "vx", "vy", "long_off", "lat_off", "ang_off", 'DSF'],
        #     },
        #     "action": {
        #         "type": "ContinuousAction",
        #         "steering_range": [-np.pi / 3, np.pi / 3],
        #         "longitudinal": True,
        #         "lateral": True,
        #         "dynamical": True
        #     },
        # })
        return config

class MultiAgentIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                 "type": "MultiAgentAction",
                 "action_config": {
                     "type": "DiscreteMetaAction",
                     "lateral": False,
                     "longitudinal": True
                 }
            },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }
            },
            "controlled_vehicles": 2
        })
        return config


TupleMultiAgentIntersectionEnv = MultiAgentWrapper(MultiAgentIntersectionEnv)
