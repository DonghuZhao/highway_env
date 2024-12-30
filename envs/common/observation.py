from collections import OrderedDict
from itertools import product
from typing import List, Dict, TYPE_CHECKING, Optional, Union, Tuple
from gymnasium import spaces
import numpy as np
import pandas as pd
import math

from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.envs.common.graphics import EnvViewer
from highway_env.road.lane import AbstractLane
from highway_env.utils import distance_to_circle, Vector
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class ObservationType(object):
    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class GrayscaleObservation(ObservationType):

    """
    An observation class that collects directly what the simulator renders.

    Also stacks the collected frames as in the nature DQN.
    The observation shape is C x W x H.

    Specific keys are expected in the configuration dictionary passed.
    Example of observation dictionary in the environment config:
        observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84)
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion,
        }
    """

    def __init__(self, env: 'AbstractEnv',
                 observation_shape: Tuple[int, int],
                 stack_size: int,
                 weights: List[float],
                 scaling: Optional[float] = None,
                 centering_position: Optional[List[float]] = None,
                 **kwargs) -> None:
        super().__init__(env)
        self.observation_shape = observation_shape
        self.shape = (stack_size, ) + self.observation_shape
        self.weights = weights
        self.obs = np.zeros(self.shape, dtype=np.uint8)

        # The viewer configuration can be different between this observation and env.render() (typically smaller)
        viewer_config = env.config.copy()
        viewer_config.update({
            "offscreen_rendering": True,
            "screen_width": self.observation_shape[0],
            "screen_height": self.observation_shape[1],
            "scaling": scaling or viewer_config["scaling"],
            "centering_position": centering_position or viewer_config["centering_position"]
        })
        self.viewer = EnvViewer(env, config=viewer_config)

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)

    def observe(self) -> np.ndarray:
        new_obs = self._render_to_grayscale()
        self.obs = np.roll(self.obs, -1, axis=0)
        self.obs[-1, :, :] = new_obs
        return self.obs

    def _render_to_grayscale(self) -> np.ndarray:
        self.viewer.observer_vehicle = self.observer_vehicle
        self.viewer.display()
        raw_rgb = self.viewer.get_image()  # H x W x C
        raw_rgb = np.moveaxis(raw_rgb, 0, 1)
        return np.dot(raw_rgb[..., :3], self.weights).clip(0, 255).astype(np.uint8)


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env: 'AbstractEnv', horizon: int = 10, **kwargs: dict) -> None:
        super().__init__(env)
        self.horizon = horizon

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(shape=self.observe().shape, low=0, high=1, dtype=np.float32)
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros((3, 3, int(self.horizon * self.env.config["policy_frequency"])))
        grid = compute_ttc_grid(self.env, vehicle=self.observer_vehicle,
                                time_quantization=1/self.env.config["policy_frequency"], horizon=self.horizon)
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.observer_vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.observer_vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0:lf+1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_speeds = 3
        v0 = grid.shape[0] + self.observer_vehicle.speed_index - obs_speeds // 2
        vf = grid.shape[0] + self.observer_vehicle.speed_index + obs_speeds // 2
        clamped_grid = padded_grid[v0:vf + 1, :, :]
        return clamped_grid.astype(np.float32)


class KinematicObservation(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = False,
                 observe_intentions: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        # print("input:", features)
        # print("default:", self.FEATURES)
        self.features = features or self.FEATURES
        self.nearby_vehicles_features = [ feature for feature in self.features].append('DSF')
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        # print("init:", self.features)
        # print("init:", self.features_range)

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)
        # Add ego-vehicle
        dict = self.observer_vehicle.to_dict()
        df = pd.DataFrame.from_records([dict])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = pd.concat([df, pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features]],
                           ignore_index=True)

        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat([df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True)
        # Reorder
        df = df[self.features]

        if self.env.config["action"]["EHMI"]:
            df["EHMI"] = 1 if self.env.vehicle.EHMI == 'Y' else 0
            df["Safety"] = close_vehicles[0].DSF if close_vehicles else 0
            # df["Safety"] = 0

        # # 判断是否在交叉口范围内
        # intersection_presence = False
        # if intersection_presence:
        #     df['presence'] = 0.
        #     df['presence'] = np.where((df['x'] > -1) & (df['x'] < 31) & (df['y'] > -32), 1., df['presence'])
        # # print(df)
        # # Normalize and clip
        # if self.normalize:
        #     df = self.normalize_obs(df)
        # print(df)
        #
        # # 是否存储相对信息
        # relative_position = False
        # if relative_position:
        #     for i in range(1, self.vehicles_count):
        #         df.iloc[i, df.columns.get_loc('x')] -= df.iloc[0, df.columns.get_loc('x')]
        #         df.iloc[i, df.columns.get_loc('y')] -= df.iloc[0, df.columns.get_loc('y')]
        #         df.iloc[i, df.columns.get_loc('vx')] -= df.iloc[0, df.columns.get_loc('vx')]
        #         df.iloc[i, df.columns.get_loc('vy')] -= df.iloc[0, df.columns.get_loc('vy')]
        # # print(df)

        ego_vehicle = df.iloc[0, :].to_dict()
        vehicles = {}
        vehicles[0] = df.iloc[1, :].to_dict()
        # print("ego_vehicle:", ego_vehicle)
        # print("vehicles:", vehicles)
        lidar_obs = ObservationUtils.process_radar_information(ego_vehicle, vehicles)
        # print(lidar_obs)

        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        ego_obs = df.iloc[0, :].values.copy()
        # obs = df.values.copy()
        # if self.order == "shuffled":
        #     self.env.np_random.shuffle(obs[1:])
        # Flatten
        # obs = obs.reshape(-1)
        obs = ego_obs.reshape(-1)
        obs = np.concatenate((obs, lidar_obs[0]))
        # print(obs)
        return obs.astype(self.space().dtype)


class OccupancyGridObservation(ObservationType):

    """Observe an occupancy grid of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'vx', 'vy', 'on_road']
    GRID_SIZE: List[List[float]] = [[-5.5*5, 5.5*5], [-5.5*5, 5.5*5]]
    GRID_STEP: List[int] = [5, 5]

    def __init__(self,
                 env: 'AbstractEnv',
                 features: Optional[List[str]] = None,
                 grid_size: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                 grid_step: Optional[Tuple[float, float]] = None,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 align_to_vehicle_axes: bool = False,
                 clip: bool = True,
                 as_image: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param grid_size: real world size of the grid [[min_x, max_x], [min_y, max_y]]
        :param grid_step: steps between two cells of the grid [step_x, step_y]
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: use absolute or relative coordinates
        :param align_to_vehicle_axes: if True, the grid axes are aligned with vehicle axes. Else, they are aligned
               with world axes.
        :param clip: clip the observation in [-1, 1]
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        self.grid_step = np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        grid_shape = np.asarray(np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step),
                                dtype=np.uint8)
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute
        self.align_to_vehicle_axes = align_to_vehicle_axes
        self.clip = clip
        self.as_image = as_image

    def space(self) -> spaces.Space:
        if self.as_image:
            return spaces.Box(shape=self.grid.shape, low=0, high=255, dtype=np.uint8)
        else:
            return spaces.Box(shape=self.grid.shape, low=-np.inf, high=np.inf, dtype=np.float32)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # Initialize empty data
            self.grid.fill(np.nan)

            # Get nearby traffic data
            df = pd.DataFrame.from_records(
                [v.to_dict(self.observer_vehicle) for v in self.env.road.vehicles])
            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                if feature in df.columns:  # A vehicle feature
                    for _, vehicle in df[::-1].iterrows():
                        x, y = vehicle["x"], vehicle["y"]
                        # Recover unnormalized coordinates for cell index
                        if "x" in self.features_range:
                            x = utils.lmap(x, [-1, 1], [self.features_range["x"][0], self.features_range["x"][1]])
                        if "y" in self.features_range:
                            y = utils.lmap(y, [-1, 1], [self.features_range["y"][0], self.features_range["y"][1]])
                        cell = self.pos_to_index((x, y), relative=not self.absolute)
                        if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                            self.grid[layer, cell[1], cell[0]] = vehicle[feature]
                elif feature == "on_road":
                    self.fill_road_layer_by_lanes(layer)

            obs = self.grid

            if self.clip:
                obs = np.clip(obs, -1, 1)

            if self.as_image:
                obs = ((np.clip(obs, -1, 1) + 1) / 2 * 255).astype(np.uint8)

            obs = np.nan_to_num(obs).astype(self.space().dtype)

            return obs

    def pos_to_index(self, position: Vector, relative: bool = False) -> Tuple[int, int]:
        """
        Convert a world position to a grid cell index

        If align_to_vehicle_axes the cells are in the vehicle's frame, otherwise in the world frame.

        :param position: a world position
        :param relative: whether the position is already relative to the observer's position
        :return: the pair (i,j) of the cell index
        """
        if not relative:
            position -= self.observer_vehicle.position
        if self.align_to_vehicle_axes:
            c, s = np.cos(self.observer_vehicle.heading), np.sin(self.observer_vehicle.heading)
            position = np.array([[c, s], [-s, c]]) @ position
        return int(np.floor((position[0] - self.grid_size[0, 0]) / self.grid_step[0])),\
               int(np.floor((position[1] - self.grid_size[1, 0]) / self.grid_step[1]))

    def index_to_pos(self, index: Tuple[int, int]) -> np.ndarray:

        position = np.array([
            (index[1] + 0.5) * self.grid_step[0] + self.grid_size[0, 0],
            (index[0] + 0.5) * self.grid_step[1] + self.grid_size[1, 0]
        ])
        if self.align_to_vehicle_axes:
            c, s = np.cos(-self.observer_vehicle.heading), np.sin(-self.observer_vehicle.heading)
            position = np.array([[c, s], [-s, c]]) @ position

        position += self.observer_vehicle.position
        return position

    def fill_road_layer_by_lanes(self, layer_index: int, lane_perception_distance: float = 100) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) information

        Here, we iterate over lanes and regularly placed waypoints on these lanes to fill the corresponding cells.
        This approach is faster if the grid is large and the road network is small.

        :param layer_index: index of the layer in the grid
        :param lane_perception_distance: lanes are rendered +/- this distance from vehicle location
        """
        lane_waypoints_spacing = np.amin(self.grid_step)
        road = self.env.road

        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for lane in road.network.graph[_from][_to]:
                    origin, _ = lane.local_coordinates(self.observer_vehicle.position)
                    waypoints = np.arange(origin - lane_perception_distance,
                                            origin + lane_perception_distance,
                                            lane_waypoints_spacing).clip(0, lane.length)
                    for waypoint in waypoints:
                        cell = self.pos_to_index(lane.position(waypoint, 0))
                        if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                            self.grid[layer_index, cell[1], cell[0]] = 1

    def fill_road_layer_by_cell(self, layer_index) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) information

        In this implementation, we iterate the grid cells and check whether the corresponding world position
        at the center of the cell is onroad/offroad. This approach is faster if the grid is small and the road network large.
        """
        road = self.env.road
        for i, j in product(range(self.grid.shape[-2]), range(self.grid.shape[-1])):
            for _from in road.network.graph.keys():
                for _to in road.network.graph[_from].keys():
                    for lane in road.network.graph[_from][_to]:
                        if lane.on_lane(self.index_to_pos((i, j))):
                            self.grid[layer_index, i, j] = 1


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env: 'AbstractEnv', scales: List[float], **kwargs: dict) -> None:
        self.scales = np.array(scales)
        super().__init__(env, **kwargs)

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float64),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float64),
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float64),
            ))
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        if not self.observer_vehicle:
            return OrderedDict([
                ("observation", np.zeros((len(self.features),))),
                ("achieved_goal", np.zeros((len(self.features),))),
                ("desired_goal", np.zeros((len(self.features),)))
            ])

        obs = np.ravel(pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features])
        goal = np.ravel(pd.DataFrame.from_records([self.env.goal.to_dict()])[self.features])
        obs = OrderedDict([
            ("observation", obs / self.scales),
            ("achieved_goal", obs / self.scales),
            ("desired_goal", goal / self.scales)
         ])
        return obs


class AttributesObservation(ObservationType):
    def __init__(self, env: 'AbstractEnv', attributes: List[str], **kwargs: dict) -> None:
        self.env = env
        self.attributes = attributes

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict({
                attribute: spaces.Box(-np.inf, np.inf, shape=obs[attribute].shape, dtype=np.float64)
                for attribute in self.attributes
            })
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        return OrderedDict([
            (attribute, getattr(self.env, attribute)) for attribute in self.attributes
        ])


class MultiAgentObservation(ObservationType):
    def __init__(self,
                 env: 'AbstractEnv',
                 observation_config: dict,
                 **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory(self.env, self.observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.agents_observation_types])

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.agents_observation_types)


class TupleObservation(ObservationType):
    def __init__(self,
                 env: 'AbstractEnv',
                 observation_configs: List[dict],
                 **kwargs) -> None:
        super().__init__(env)
        self.observation_types = [observation_factory(self.env, obs_config) for obs_config in observation_configs]

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.observation_types])

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.observation_types)


class ExitObservation(KinematicObservation):

    """Specific to exit_env, observe the distance to the next exit lane as part of a KinematicObservation."""

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        ego_dict = self.observer_vehicle.to_dict()
        exit_lane = self.env.road.network.get_lane(("1", "2", -1))
        ego_dict["x"] = exit_lane.local_coordinates(self.observer_vehicle.position)[0]
        df = pd.DataFrame.from_records([ego_dict])[self.features]

        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind)
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = pd.concat([df, pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features]],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat([df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs


class LidarObservation(ObservationType):
    DISTANCE = 0
    SPEED = 1

    def __init__(self, env,
                 cells: int = 16,
                 maximum_range: float = 60,
                 normalize: bool = True,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.cells = cells
        self.maximum_range = maximum_range
        self.normalize = normalize
        self.angle = 2 * np.pi / self.cells
        self.grid = np.ones((self.cells, 1)) * float('inf')
        self.origin = None

    def space(self) -> spaces.Space:
        high = 1 if self.normalize else self.maximum_range
        return spaces.Box(shape=(self.cells, 2), low=-high, high=high, dtype=np.float32)

    def observe(self) -> np.ndarray:
        obs = self.trace(self.observer_vehicle.position, self.observer_vehicle.velocity).copy()
        if self.normalize:
            obs /= self.maximum_range
        return obs

    def trace(self, origin: np.ndarray, origin_velocity: np.ndarray) -> np.ndarray:
        self.origin = origin.copy()
        self.grid = np.ones((self.cells, 2)) * self.maximum_range

        for obstacle in self.env.road.vehicles + self.env.road.objects:
            if obstacle is self.observer_vehicle or not obstacle.solid:
                continue
            center_distance = np.linalg.norm(obstacle.position - origin)
            if center_distance > self.maximum_range:
                continue
            center_angle = self.position_to_angle(obstacle.position, origin)
            center_index = self.angle_to_index(center_angle)
            distance = center_distance - obstacle.WIDTH / 2
            if distance <= self.grid[center_index, self.DISTANCE]:
                direction = self.index_to_direction(center_index)
                velocity = (obstacle.velocity - origin_velocity).dot(direction)
                self.grid[center_index, :] = [distance, velocity]

            # Angular sector covered by the obstacle
            corners = utils.rect_corners(obstacle.position, obstacle.LENGTH, obstacle.WIDTH, obstacle.heading)
            angles = [self.position_to_angle(corner, origin) for corner in corners]
            min_angle, max_angle = min(angles), max(angles)
            start, end = self.angle_to_index(min_angle), self.angle_to_index(max_angle)
            if start < end:
                indexes = np.arange(start, end+1)
            else:
                indexes = np.hstack([np.arange(start, self.cells), np.arange(0, end + 1)])

            # Actual distance computation for these sections
            for index in indexes:
                direction = self.index_to_direction(index)
                ray = [origin, origin + self.maximum_range * direction]
                distance = utils.distance_to_rect(ray, corners)
                if distance <= self.grid[index, self.DISTANCE]:
                    velocity = (obstacle.velocity - origin_velocity).dot(direction)
                    self.grid[index, :] = [distance, velocity]
        return self.grid

    def position_to_angle(self, position: np.ndarray, origin: np.ndarray) -> float:
        return np.arctan2(position[1] - origin[1], position[0] - origin[0]) + self.angle/2

    def position_to_index(self, position: np.ndarray, origin: np.ndarray) -> int:
        return self.angle_to_index(self.position_to_angle(position, origin))

    def angle_to_index(self, angle: float) -> int:
        return int(np.floor(angle / self.angle)) % self.cells

    def index_to_direction(self, index: int) -> np.ndarray:
        return np.array([np.cos(index * self.angle), np.sin(index * self.angle)])


def observation_factory(env: 'AbstractEnv', config: dict) -> ObservationType:
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, **config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    elif config["type"] == "TupleObservation":
        return TupleObservation(env, **config)
    elif config["type"] == "LidarObservation":
        return LidarObservation(env, **config)
    elif config["type"] == "ExitObservation":
        return ExitObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")


class ObservationUtils:

    @staticmethod
    def radians_to_degrees(radians):
        """将角度从弧度单位转换为度单位。

        Args:
        - radians : float
            角度，以弧度为单位。

        Returns:
        - degrees : float
            角度，以度为单位。
            """
        degrees = radians * (180 / math.pi)
        return degrees

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        """计算两点之间的距离。

        Args:
            x1, y1, x2, y2: 两点全局坐标

        Returns:
            绝对距离
        """
        return np.float32(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    @staticmethod
    def calculate_angle(main_car_pos, main_car_heading, other_car_pos):
        """
        计算其他车辆相对于主车前进方向的角度。

        参数:
        main_car_pos (tuple): 主车的位置 (x, y)
        main_car_heading (float): 主车的航向角度，以度为单位
        other_car_pos (tuple): 其他车辆的位置 (x, y)

        返回:
        float: 其他车辆相对于主车前进方向的角度（度）
        """
        # 将主车航向角从[-2π, 0]转换到[0, 2π]
        main_car_heading = - main_car_heading

        # 计算向量AB
        vector_AB = ([other_car_pos[0] - main_car_pos[0], other_car_pos[1] - main_car_pos[1]])

        # 东向向量
        vector_east = (1, 0)

        # 计算点积和叉积
        dot = vector_east[0] * vector_AB[0] + vector_east[1] * vector_AB[1]
        cross = - (vector_AB[1] * vector_east[0] - vector_AB[0] * vector_east[1])

        # 计算角度
        angle = np.arctan2(cross, dot)
        angle_degrees = np.degrees(angle)

        # 调整角度到[0, 360)
        if angle_degrees < 0:
            angle_degrees += 360

        ego_yaw_degrees = ObservationUtils.radians_to_degrees(main_car_heading)
        angle_diff = angle_degrees - ego_yaw_degrees

        if angle_diff < 0:
            angle_diff += 360

        return angle_diff

    @staticmethod
    def adjust_v_yaw(yaw_rad):
        """
        转换车辆的航向角坐标系,使其保持在逆时针为[-360,0]
                -270/90
                    |
        -180/180 ——   —— 0
                    |
                -90
        """
        # 针对小于-360度的情况
        if yaw_rad < -2 * math.pi:
            yaw_rad = yaw_rad + 2 * math.pi

        # 针对大于0的情况
        if yaw_rad > 0:
            yaw_rad = -2 * math.pi + yaw_rad

        # 判断yaw_rad是否在[-2pi,0]区间，如果不是则唤起错误
        if yaw_rad < -2 * math.pi or yaw_rad > 0:
            raise ValueError("yaw_rad is not in the range of [-2pi, 0]")

        return yaw_rad
    
    @staticmethod
    def process_radar_information(ego_vehicle, vehicles):
        """处理并返回所有车辆的雷达信息。

        Args:
            vehicles (dict): 车辆字典,键为车辆ID,值为车辆状态对象。

        Returns:
            dict: 每个车辆ID对应的雷达观测数据。
        """
        all_v_ids = list(vehicles.keys())
        num_agents = len(all_v_ids)

        glob_x = []
        glob_y = []
        glob_v = []
        glob_yaw = []

        # 提取所有车辆的位置坐标
        for agent_id in all_v_ids:
            x = vehicles[agent_id]['x']
            y = vehicles[agent_id]['y']
            yaw = vehicles[agent_id]['heading']
            yaw = ObservationUtils.adjust_v_yaw(yaw)
            # print('rad:', yaw)
            # print('degree:', radians_to_degrees(yaw))
            v = vehicles[agent_id]['speed']
            glob_x.append(np.float32(x))
            glob_y.append(np.float32(y))
            glob_v.append(np.float32(v))
            glob_yaw.append(np.float32(yaw))

        glob_x = np.array(glob_x)
        glob_y = np.array(glob_y)
        glob_v = np.array(glob_v)
        glob_yaw = np.array(glob_yaw)

        #  最终返回的雷达信息和冲突信息
        lidar_obs = {}

        ego_position = (ego_vehicle['x'], ego_vehicle['y'])
        # 计算每个车辆与主车的距离
        for agent_id in range(num_agents):
            agent_position = (glob_x[agent_id], glob_y[agent_id])

            distance = ObservationUtils.calculate_distance(ego_vehicle['x'], ego_vehicle['y'],
                                                           glob_x[agent_id], glob_y[agent_id])

            angle = ObservationUtils.calculate_angle(ego_position, ego_vehicle['heading'], agent_position)

            # -------------仅输出distance,angle,speed------------
            # 归一化距离
            max_distance = np.float32(60.0)
            norm_distance = min(distance / max_distance, 1.0)
            norm_angle = min(angle / 360, 1)
            speed = min(glob_v[agent_id] / 10, 1)
            lidar = [norm_distance, norm_angle, speed]
            lidar_obs[agent_id] = lidar

            # # ------------输出雷达结果---------------------------
            # # 固定信息到对应角度
            # lidars = ObservationUtils.update_lidar_with_closest_cars([distance], [angle], n_interval=8)
            #
            # # 归一化距离
            # max_distance = np.float32(60.0)
            #
            # norm_lidar = [min(lidar / max_distance, 1.0) for lidar in lidars]
            # agent_lidar = np.array(norm_lidar, dtype=np.float32)
            # lidar_obs[agent_id] = agent_lidar
            # # print('lidar:', len(agent_lidar), agent_lidar)

        return lidar_obs

    @staticmethod
    def update_lidar_with_closest_cars(distances, angles,n_interval: int = 36):
        """
        更新lidar列表，根据车辆的距离和角度信息，并只保留每个区间内距离最近的车辆。

        参数:
        distances (list): 每辆车相对于主车的距离列表。
        angles (list): 每辆车相对于主车的角度列表。

        返回:
        list: 更新后的lidar列表。
        """
        # 初始化lidar列表，所有值设为1000
        lidar = [1000] * n_interval
        interval = 360 / n_interval
        # 计算每辆车所属的区间，并将距离最近的车辆ID放入对应的区间
        for car_id, (distance, angle) in enumerate(zip(distances, angles), start=1):
            # 计算对应的区间索引
            index = int(angle // interval)
            # 由于区间是从0到35，我们需要将360度归类到第0个元素
            if index == n_interval:
                index = 0
            # 确保index在范围内
            if index >= 0 and index < n_interval:
                current_car_dist = lidar[index]
            else:
                print("Index out of range:", index)
                index = n_interval - 1
                current_car_dist = lidar[index]
                # continue
            # print('angle:', angle)
            if current_car_dist == 1000 or distance < distances[current_car_dist - 1]:
                lidar[index] = car_id

        # 将车辆ID替换为对应的距离值
        for i in range(len(lidar)):
            if lidar[i] != 1000:  # 说明有车辆id
                # 获取车辆ID对应的距离
                lidar[i] = distances[lidar[i] - 1]  # car_id是从1开始的，而索引是从0开始的

        return lidar