import numpy as np
import pandas as pd
import math

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
    def update_lidar_with_closest_cars(distances, angles, n_interval: int = 36):
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