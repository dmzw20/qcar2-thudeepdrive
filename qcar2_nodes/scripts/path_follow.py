#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Path
from qcar2_interfaces.msg import MotorCommands
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import math
from tf2_ros import TransformException # type: ignore
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import time
from collections import deque
import numpy as np
import cv2
import pickle
import yaml
import os
from cv_bridge import CvBridge # type: ignore

# 从 path_to_nav.py 复制 Road 类定义，以便加载 pickle 文件
class Road:
    def __init__(self, id, points, name=""):
        self.id = id
        self.points = points  # 道路点列表 [(y1, x1), (y2, x2),...]
        self.name = name if name else f"Road {id}"

class PathFollowController(Node):
    """ROS2节点：路径跟随控制器，进行纯路径跟踪导航并支持多种导航模式"""
    
    def __init__(self):
        super().__init__('path_follow_controller')
        
        # --- 常量定义 ---
        self.LOOKAHEAD_DISTANCE = 0.3  # 前瞻距离（米）
        self.MAX_SPEED = 0.2  # 最大速度（米/秒）
        self.MIN_SPEED = 0.05  # 最小速度（米/秒）
        self.MAX_STEERING = 0.6  # 最大转向角（弧度）
        
        # --- 声明参数 ---
        self.declare_parameter('road_definition_file', 'roads_definition.pkl')
        self.declare_parameter('transform_matrix_file', 'map_transform.yaml')
        self.declare_parameter('lookahead_distance_ros', 0.45)  # ROS坐标系下的前瞻距离 (米)
        self.declare_parameter('loop_road_id', 5)  # 'loop' 道路的 ID
        self.declare_parameter('publish_nav_commands', True)  # 是否发布导航命令
        self.declare_parameter('nav_command_topic', '/nav_command')  # 导航命令话题
        
        # --- 红绿灯状态相关参数 ---
        self.declare_parameter('traffic_light_topic', '/traffic_light_state')  # 红绿灯状态话题
        self.traffic_light_topic = self.get_parameter('traffic_light_topic').get_parameter_value().string_value
        
        # --- 红绿灯状态相关变量 ---
        self.traffic_light_state = 'none'  # 当前红绿灯状态
        self.path_follow_blocked = False  # 路径跟踪是否被红灯阻止
        self.mode_just_changed_to_path_follow = False  # 是否刚刚切换到路径跟踪模式
        
        # --- 环道跟踪参数 ---
        # 图像处理参数
        self.declare_parameter('loop_crop_height_start', 580 // 2)  # 环道图像裁剪起始高度
        self.declare_parameter('loop_crop_height_end', 818 // 2)    # 环道图像裁剪结束高度
        self.declare_parameter('loop_crop_width', 820)             # 环道图像裁剪宽度
        # 颜色阈值参数
        self.declare_parameter('loop_hsv_lower_white', [0, 0, 200])    # 环道白色HSV下限
        self.declare_parameter('loop_hsv_upper_white', [180, 30, 255]) # 环道白色HSV上限
        self.declare_parameter('loop_hsv_lower_black', [0, 0, 0])      # 环道黑色HSV下限
        self.declare_parameter('loop_hsv_upper_black', [180, 255, 70]) # 环道黑色HSV上限
        # 控制参数
        self.declare_parameter('loop_target_speed', 0.15)          # 环道目标速度
        self.declare_parameter('loop_steering_gain', 0.45)         # 环道转向增益
        self.declare_parameter('loop_speed_reduction', 0.6)        # 环道转弯减速因子
        self.declare_parameter('loop_max_steering', 0.5)           # 环道最大转向角
        self.declare_parameter('loop_image_center_offset', 62)     # 环道图像中心点偏移（像素）
        # 线检测参数
        self.declare_parameter('loop_min_line_points', 3)          # 环道最小线点数
        self.declare_parameter('loop_sample_points', 10)           # 环道采样点数量
        self.declare_parameter('loop_dilation_kernel', 40)         # 环道膨胀内核大小
        self.declare_parameter('loop_poly_order', 2)               # 环道多项式拟合阶数
        self.declare_parameter('loop_lane_width', 300)             # 环道估计车道宽度（像素）
        self.declare_parameter('loop_black_divider_points', 50)    # 环道黑色分界点数量
        # 其他参数
        self.declare_parameter('loop_delay_start', 1.5)            # 环道跟踪启动延迟(秒)
        
        self.declare_parameter('turn_around_enabled', True)  # 是否启用掉头功能
        self.declare_parameter('min_curvature_radius', 0.20)  # 掉头判断的最小曲率半径（米）
        self.declare_parameter('min_angle_change', 80.0)    # 掉头判断的最小角度变化（度）
        self.declare_parameter('turn_around_detection_distance', 0.5)  # 掉头检测的前方距离（米）
        
        # --- 获取掉头相关参数 ---
        self.turn_around_enabled = self.get_parameter('turn_around_enabled').get_parameter_value().bool_value
        self.min_curvature_radius = self.get_parameter('min_curvature_radius').get_parameter_value().double_value
        self.min_angle_change = math.radians(self.get_parameter('min_angle_change').get_parameter_value().double_value)  # 转换为弧度
        self.turn_around_detection_distance = self.get_parameter('turn_around_detection_distance').get_parameter_value().double_value
        
        # --- 掉头状态相关变量 ---
        self.turn_around_in_progress = False       # 是否正在执行掉头
        self.turn_around_start_time = None         # 掉头开始时间
        self.turn_around_step = 0                  # 掉头步骤
        self.turn_around_last_command_time = None  # 上次掉头命令发送时间
        self.just_completed_turn_around = False
        self.turn_around_completion_time = 0
        
        # --- 掉头控制参数 (参考turn_around.py) ---
        self.turn_around_max_steering = 0.6        # 掉头最大转向角度（弧度）
        self.turn_around_normal_speed = 0.2        # 掉头正常速度 (m/s)
        self.turn_around_slow_speed = 0.15         # 掉头慢速 (m/s)
        
        # --- 获取参数 ---
        road_file = self.get_parameter('road_definition_file').get_parameter_value().string_value
        transform_file = self.get_parameter('transform_matrix_file').get_parameter_value().string_value
        self.lookahead_distance_ros = self.get_parameter('lookahead_distance_ros').get_parameter_value().double_value
        self.loop_road_id = self.get_parameter('loop_road_id').get_parameter_value().integer_value
        self.publish_nav_commands = self.get_parameter('publish_nav_commands').get_parameter_value().bool_value
        nav_command_topic = self.get_parameter('nav_command_topic').get_parameter_value().string_value
        
        # --- 获取环道跟踪参数 ---
        # 图像处理参数
        self.loop_crop_h_start = self.get_parameter('loop_crop_height_start').value
        self.loop_crop_h_end = self.get_parameter('loop_crop_height_end').value
        self.loop_crop_w = self.get_parameter('loop_crop_width').value
        # 颜色阈值参数
        self.loop_white_lower = np.array(self.get_parameter('loop_hsv_lower_white').get_parameter_value().integer_array_value, dtype=np.uint8)
        self.loop_white_upper = np.array(self.get_parameter('loop_hsv_upper_white').get_parameter_value().integer_array_value, dtype=np.uint8)
        self.loop_black_lower = np.array(self.get_parameter('loop_hsv_lower_black').get_parameter_value().integer_array_value, dtype=np.uint8)
        self.loop_black_upper = np.array(self.get_parameter('loop_hsv_upper_black').get_parameter_value().integer_array_value, dtype=np.uint8)
        # 控制参数
        self.loop_target_speed = self.get_parameter('loop_target_speed').value
        self.loop_steering_gain = self.get_parameter('loop_steering_gain').value
        self.loop_speed_reduction = self.get_parameter('loop_speed_reduction').value
        self.loop_max_steering = self.get_parameter('loop_max_steering').value
        self.loop_image_center_offset = self.get_parameter('loop_image_center_offset').value
        # 线检测参数
        self.loop_min_points = self.get_parameter('loop_min_line_points').value
        self.loop_sample_points = self.get_parameter('loop_sample_points').value
        self.loop_dilation_size = self.get_parameter('loop_dilation_kernel').value
        self.loop_poly_order = self.get_parameter('loop_poly_order').value
        self.loop_lane_width = self.get_parameter('loop_lane_width').value
        self.loop_black_divider_points = self.get_parameter('loop_black_divider_points').value
        # 其他参数
        self.loop_delay_start = self.get_parameter('loop_delay_start').value
        
        # --- 初始化变量 ---
        self.path = None  # 规划路径
        self.current_path_index = 0  # 当前路径索引
        self.last_control_time = time.time()  # 上次控制时间
        self.steering_angle = 0.0  # 当前转向角
        self.motor_throttle = 0.0  # 当前油门
        
        # --- 道路检测相关初始化 ---
        self.roads = self.load_roads(road_file)
        self.transformation_matrix = self.load_transformation_matrix(transform_file)
        self.inv_transformation_matrix = None
        if self.transformation_matrix is not None:
            try:
                self.inv_transformation_matrix = np.linalg.inv(self.transformation_matrix)
            except np.linalg.LinAlgError:
                self.get_logger().error("无法计算逆变换矩阵")
                self.inv_transformation_matrix = None
        
        self.current_pose = None  # 当前位姿
        self.current_pixel_pos = None  # 当前像素位置
        self.current_yaw = None  # 当前朝向(弧度)
        self.current_nav_mode = "follow path"  # 当前导航模式
        self.prev_nav_mode = None  # 上一个导航模式
        self.mode_change_time = None  # 模式变更时间
        
        # --- 创建TF监听器 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # --- 创建订阅器 ---
        self.path_subscription = self.create_subscription(
            Path,
            '/planned_path',
            self.path_callback,
            10)
        
        self.traffic_light_subscription = self.create_subscription(
            String,
            self.traffic_light_topic,
            self.traffic_light_callback,
            10)
        
        # --- 创建发布器 ---
        self.motor_publisher = self.create_publisher(
            MotorCommands,
            '/qcar2_motor_speed_cmd',
            10)
        
        # --- 创建导航命令发布器 ---
        if self.publish_nav_commands:
            self.nav_command_publisher = self.create_publisher(
                String,
                nav_command_topic,
                10)
        
        # --- 创建定时器，定期执行控制 ---
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        
        # --- 过滤器队列，用于平滑控制 ---
        self.filter_size = 3
        self.speed_queue = deque(maxlen=self.filter_size)
        self.steering_queue = deque(maxlen=self.filter_size)
        
        # --- 环道跟踪内部变量 ---
        self.loop_last_steering = 0.0    # 上一次环道转向命令
        self.loop_integral_error = 0.0   # 环道积分误差
        self.loop_last_error = 0.0       # 环道上一次误差
        self.loop_max_integral = 0.5     # 环道积分上限
        self.loop_last_time = self.get_clock().now()  # 环道最后控制时间
        
        self.previous_road_id = None  # 记录切换到环道前的道路ID
        self.loop_entry_strategies = {
            # 根据不同道路ID设置不同的环道入口策略
            # 道路ID: (初始转向角, 初始速度, 延迟时间, 图像中心偏移)
            2: (-0.05, 0.2, 4.3, 62),   # 道路2是直线进入环道
            3: (0.25, 0.15, 3.0, 62),   # 道路3是右转进入环道 TO BE TESTED
            4: (0.25, 0.15, 3.0, 62),   # 道路4是右转进入环道 TO BE TESTED
            # 默认策略
            "default": (-0.05, 0.15, 1.5, 62)
        }
        
        # --- 黄线跟踪参数 ---
        # 图像处理参数
        self.declare_parameter('crop_height_start', 630 // 2) # 图像裁剪起始高度
        self.declare_parameter('crop_height_end', 818 // 2)   # 图像裁剪结束高度
        self.declare_parameter('crop_width', 820)            # 图像裁剪宽度
        self.declare_parameter('hsv_lower_yellow', [10, 50, 100]) # 黄色HSV下限
        self.declare_parameter('hsv_upper_yellow', [45, 255, 255]) # 黄色HSV上限
        # 添加黑色和白色的HSV范围参数
        self.declare_parameter('hsv_lower_black', [0, 0, 0])      # 黑色HSV下限
        self.declare_parameter('hsv_upper_black', [180, 255, 70]) # 黑色HSV上限
        self.declare_parameter('hsv_lower_white', [0, 0, 200])    # 白色HSV下限
        self.declare_parameter('hsv_upper_white', [180, 30, 255]) # 白色HSV上限
        # 控制参数
        self.declare_parameter('target_speed', 0.25)         # 目标速度
        self.declare_parameter('steering_gain', 0.45)         # 默认转向增益
        self.declare_parameter('max_steering', 0.5)          # 最大转向角
        self.declare_parameter('speed_reduction_factor', 0.6) # 默认转弯减速因子
        # 添加左转和右转特定参数
        self.declare_parameter('left_turn_steering_gain', 0.45)   # 左转专用转向增益
        self.declare_parameter('right_turn_steering_gain', 0.5)   # 右转专用转向增益
        self.declare_parameter('left_turn_speed_reduction', 0.5)  # 左转速度减速因子
        self.declare_parameter('right_turn_speed_reduction', 0.5) # 右转速度减速因子
        self.declare_parameter('turn_detection_threshold', 0.175) # 判断转向方向的阈值
        # 添加多项式拟合相关参数
        self.declare_parameter('poly_order', 2)              # 多项式拟合阶数
        self.declare_parameter('sample_points', 10)          # 采样点数量
        self.declare_parameter('min_points', 3)              # 最小拟合点数
        # 添加膨胀操作参数
        self.declare_parameter('yellow_dilation_kernel', 40)  # 黄色掩码膨胀内核大小
        self.declare_parameter('white_dilation_kernel', 40)   # 白色掩码膨胀内核大小
        # 添加黄线跟随偏移量参数
        self.declare_parameter('yellow_line_offset', 250)     # 黄线跟随偏移量（像素）
        # 添加估计车道宽度参数
        self.declare_parameter('estimated_lane_width', 300)   # 估计车道宽度（像素）
        
        # --- 获取黄线和环道跟踪参数 ---
        # 图像处理参数
        self.crop_h_start = self.get_parameter('crop_height_start').value
        self.crop_h_end = self.get_parameter('crop_height_end').value
        self.crop_w = self.get_parameter('crop_width').value
        self.yellow_hsv_lower = np.array(self.get_parameter('hsv_lower_yellow').get_parameter_value().integer_array_value, dtype=np.uint8)
        self.yellow_hsv_upper = np.array(self.get_parameter('hsv_upper_yellow').get_parameter_value().integer_array_value, dtype=np.uint8)
        self.black_hsv_lower = np.array(self.get_parameter('hsv_lower_black').get_parameter_value().integer_array_value, dtype=np.uint8)
        self.black_hsv_upper = np.array(self.get_parameter('hsv_upper_black').get_parameter_value().integer_array_value, dtype=np.uint8)
        self.white_hsv_lower = np.array(self.get_parameter('hsv_lower_white').get_parameter_value().integer_array_value, dtype=np.uint8)
        self.white_hsv_upper = np.array(self.get_parameter('hsv_upper_white').get_parameter_value().integer_array_value, dtype=np.uint8)
        # 控制参数
        self.target_speed = self.get_parameter('target_speed').value
        self.steering_gain = self.get_parameter('steering_gain').value
        self.max_steering = self.get_parameter('max_steering').value
        self.speed_reduction = self.get_parameter('speed_reduction_factor').value
        # 左转和右转特定参数
        self.left_gain = self.get_parameter('left_turn_steering_gain').value
        self.right_gain = self.get_parameter('right_turn_steering_gain').value
        self.left_reduction = self.get_parameter('left_turn_speed_reduction').value
        self.right_reduction = self.get_parameter('right_turn_speed_reduction').value
        self.turn_threshold = self.get_parameter('turn_detection_threshold').value
        # 多项式拟合参数
        self.poly_order = self.get_parameter('poly_order').value
        self.sample_points = self.get_parameter('sample_points').value
        self.min_points = self.get_parameter('min_points').value
        # 膨胀操作参数
        self.yellow_dilation_size = self.get_parameter('yellow_dilation_kernel').value
        self.white_dilation_size = self.get_parameter('white_dilation_kernel').value
        # 偏移量参数
        self.yellow_line_offset = self.get_parameter('yellow_line_offset').value
        # 车道宽度参数
        self.est_lane_width = self.get_parameter('estimated_lane_width').value
        # 环道参数
        self.loop_target_speed = self.get_parameter('loop_target_speed').get_parameter_value().double_value
        self.loop_steering_gain = self.get_parameter('loop_steering_gain').get_parameter_value().double_value
        self.loop_delay_start = self.get_parameter('loop_delay_start').get_parameter_value().double_value
        
        # --- yellow_follow.py中使用的变量 ---
        self.last_steering = 0.0    # 上一次的转向命令，用于平滑
        self.last_measured_lane_width = self.est_lane_width  # 初始化为估计值
        self.max_measured_lane_width = self.est_lane_width   # 初始化为估计值
        self.last_dual_steering = 0.0  # 记录最后一次双线模式下的转向值
        self.right_turn_single_mode = False  # 是否处于右转单线固定转向模式
        self.consecutive_right_turns = 0  # 连续右转计数
        self.integral_error = 0.0  # 累积误差(I项)
        self.last_error = 0.0      # 上一次的误差
        self.max_integral = 0.5    # 积分项上限，防止积分饱和
        self.last_time = self.get_clock().now()  # 最后控制时间
        
        # --- 路径跟踪PID控制参数 ---
        self.cross_track_error = 0.0
        self.cross_track_error_prev = 0.0
        self.cross_track_error_sum = 0.0
        self.cross_track_error_last_time = time.time()

        # PID控制器参数
        self.kp_cross = 0.40  # 比例增益
        self.ki_cross = 0.05  # 积分增益
        self.kd_cross = 0.10  # 微分增益

        # Stanley控制器参数
        self.k_stanley = 0.5  # Stanley增益
        
        # --- 图像处理相关 ---
        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/csi_image',  # 图像话题
            self.image_callback,
            10)
        
        self.latest_image = None  # 最新图像
        self.loop_mode_active = False  # 环道跟踪模式是否激活
        
        self.get_logger().info('多模式路径跟随控制器已启动')
        if not self.roads:
            self.get_logger().warning("道路定义未加载!")
        if self.transformation_matrix is None or self.inv_transformation_matrix is None:
            self.get_logger().warning("坐标变换矩阵未加载!")
    
    def load_roads(self, file_path):
        """加载道路定义"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    roads = pickle.load(f)
                self.get_logger().info(f"已加载 {len(roads)} 条道路定义从 {file_path}")
                # 打印道路名称和ID以供参考
                for road_id, road in roads.items():
                    self.get_logger().info(f"  - Road ID: {road.id}, Name: '{road.name}', Points: {len(road.points)}")
                return roads
            else:
                self.get_logger().error(f"未找到道路定义文件: {file_path}")
                return {}
        except Exception as e:
            self.get_logger().error(f"加载道路定义失败: {e}")
            return {}

    def load_transformation_matrix(self, file_path):
        """从YAML文件加载转换矩阵"""
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            matrix = np.array(data['transformation_matrix'])
            self.get_logger().info(f"已加载坐标转换矩阵从 {file_path}")
            return matrix
        except Exception as e:
            self.get_logger().error(f"加载转换矩阵失败: {e}")
            return None
    
    def ros_to_pixel(self, ros_x, ros_y):
        """将ROS坐标转换为地图像素坐标 (y, x)"""
        if self.inv_transformation_matrix is None:
            return None

        ros_point = np.array([[[ros_x, ros_y]]], dtype=np.float32)
        try:
            pixel_point = cv2.perspectiveTransform(ros_point, self.inv_transformation_matrix)
            # 从OpenCV坐标(x,y)转换为内部使用的(y,x)格式
            x, y = pixel_point[0][0]
            return (int(round(y)), int(round(x)))
        except Exception as e:
            self.get_logger().error(f"ROS 到像素坐标转换失败: {e}")
            return None
    
    def find_road_at_pixel(self, pixel_point):
        """使用射线投射算法查找给定像素点所在的道路ID (多边形内部)"""
        if not self.roads or pixel_point is None:
            return None

        py, px = pixel_point

        for road_id, road in self.roads.items():
            if len(road.points) < 3: continue # 多边形至少需要3个顶点

            points = road.points # 格式 [(y1, x1), (y2, x2),...]
            num_vertices = len(points)
            is_inside = False

            # 射线投射算法
            p1y, p1x = points[0]
            for i in range(num_vertices + 1): # 检查所有边，包括最后一个顶点到第一个顶点的边
                p2y, p2x = points[i % num_vertices] # 获取下一个顶点，循环连接

                # 检查点是否与顶点重合
                if py == p1y and px == p1x:
                    return road_id # 点在顶点上，算作内部

                # 检查射线是否与水平边重合或穿过顶点 (特殊情况处理)
                # 如果点 Y 坐标在线段 Y 范围内
                if min(p1y, p2y) < py <= max(p1y, p2y):
                    # 并且点 X 坐标小于线段两端点的 X 最大值 (确保射线方向)
                    if px <= max(p1x, p2x):
                        # 如果线段不是水平的
                        if p1y != p2y:
                            # 计算水平射线与线段的交点 X 坐标
                            x_intersection = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            # 如果交点在点的右侧或重合
                            if x_intersection >= px:
                                is_inside = not is_inside
                        # 如果线段是水平的，并且点在线段上
                        elif p1x == p2x and py == p1y:
                             # 水平射线与水平线段重合，若点在线段上则算内部（或边界）
                             if min(p1x, p2x) <= px <= max(p1x, p2x):
                                 return road_id # 点在水平边界上

                # 更新到下一条边
                p1y, p1x = p2y, p2x

            if is_inside:
                return road_id

        return None
    
    def path_callback(self, msg):
        """接收规划路径的回调函数"""
        if not msg.poses:
            self.get_logger().warning('接收到的路径为空')
            return
            
        self.path = msg
        self.current_path_index = 0
        self.get_logger().info(f'接收到新路径，包含 {len(msg.poses)} 个点')
    
    def image_callback(self, msg):
        """处理接收到的图像消息"""
        try:
            # 将ROS图像消息转换为OpenCV图像
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'无法转换图像: {e}')
    
    def traffic_light_callback(self, msg):
        """处理接收到的红绿灯状态消息"""
        new_state = msg.data
        
        # 记录之前的状态，以便检测状态变化
        old_state = self.traffic_light_state
        self.traffic_light_state = new_state
        
        # 记录状态变化
        if new_state != old_state:
            self.get_logger().info(f'红绿灯状态改变: {old_state} -> {new_state}')
            
            # 如果状态从红灯或黄灯变为绿灯，解除路径跟踪阻塞
            if (old_state in ['red', 'yellow']) and new_state == 'green':
                self.path_follow_blocked = False
                self.get_logger().info('红绿灯变为绿色，解除路径跟踪阻塞')
    
    def get_current_pose(self):
        """获取当前车辆位置 - 使用base_scan而非base_link"""
        try:
            # 查询从map到base_scan的变换
            trans = self.tf_buffer.lookup_transform(
                'map',
                'base_scan',  # 按要求使用base_scan
                Time())
            
            # 提取位置
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            
            # 从四元数中获取yaw角度
            q = trans.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            
            # 更新当前位姿信息
            self.current_pose = (x, y, yaw)
            self.current_yaw = yaw
            
            # 转换到像素坐标
            self.current_pixel_pos = self.ros_to_pixel(x, y)
            
            return (x, y, yaw)
        
        except TransformException as ex:
            self.get_logger().warning(f'无法获取变换: {ex}')
            self.current_pose = None
            self.current_pixel_pos = None
            self.current_yaw = None
            return None
    
    def find_target_point(self, current_pose, new_lookahead_distance=None):
        """在路径上找到前瞻点"""
        if not self.path or not current_pose:
            return None
        
        lookahead_distance = self.lookahead_distance_ros
        
        if new_lookahead_distance is not None:
            lookahead_distance = new_lookahead_distance
            
        x, y, _ = current_pose
        
        # 从当前索引开始寻找前瞻点
        min_dist = float('inf')
        target_index = self.current_path_index
        
        # 首先找到路径上最近的点
        for i in range(self.current_path_index, len(self.path.poses)):
            pose = self.path.poses[i]
            dx = pose.pose.position.x - x
            dy = pose.pose.position.y - y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist:
                min_dist = dist
                self.current_path_index = i
        
        # 然后从最近点开始找到满足前瞻距离的点
        for i in range(self.current_path_index, len(self.path.poses)):
            pose = self.path.poses[i]
            dx = pose.pose.position.x - x
            dy = pose.pose.position.y - y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist >= lookahead_distance:
                target_index = i
                break
                
            # 如果已经是最后一个点
            if i == len(self.path.poses) - 1:
                target_index = i
        
        # 返回目标点
        return self.path.poses[target_index].pose.position
    
    def check_turn_around_condition(self, current_pose):
        """
        根据前方路径的曲率和角度变化判断是否需要掉头
        
        Args:
            current_pose: 当前位置 (x, y, yaw)
                
        Returns:
            bool: 是否需要掉头
        """
        
        if self.traffic_light_state in ['red', 'yellow']:
            self.get_logger().info(f'因红绿灯状态({self.traffic_light_state})而暂缓掉头操作')
            return False
        
        if not self.turn_around_enabled or not self.path or not current_pose:
            return False
        
        # 已经在执行掉头，继续执行
        if self.turn_around_in_progress:
            return True
            
        # 如果刚刚完成掉头，添加一个冷却期
        current_time = time.time()
        turn_around_cooldown = 3.0  # 掉头冷却时间（秒）
        if hasattr(self, 'just_completed_turn_around') and self.just_completed_turn_around:
            if current_time - self.turn_around_completion_time < turn_around_cooldown:
                return False
            else:
                # 冷却期结束后重置标志
                self.just_completed_turn_around = False
        
        x, y, _ = current_pose
        
        # 获取路径上的点，用于评估曲率
        path_points = []
        distances = []
        
        # 从当前位置最近的路径点开始
        min_dist = float('inf')
        nearest_idx = 0
        
        # 查找距离当前位置最近的路径点
        for i, pose in enumerate(self.path.poses):
            pose_x = pose.pose.position.x
            pose_y = pose.pose.position.y
            dist = math.sqrt((pose_x - x)**2 + (pose_y - y)**2)
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # 收集前方一定距离内的路径点
        cumulative_distance = 0.0
        last_x, last_y = x, y
        
        for i in range(nearest_idx, len(self.path.poses)):
            pose = self.path.poses[i]
            pose_x = pose.pose.position.x
            pose_y = pose.pose.position.y
            
            # 计算点间距离
            segment_dist = math.sqrt((pose_x - last_x)**2 + (pose_y - last_y)**2)
            cumulative_distance += segment_dist
            
            # 添加点坐标
            path_points.append((pose_x, pose_y))
            distances.append(cumulative_distance)
            
            last_x, last_y = pose_x, pose_y
            
            # 超出检测距离退出循环
            if cumulative_distance > self.turn_around_detection_distance:
                break
        
        # 如果点太少，无法评估
        if len(path_points) < 40:  # 需要至少10个点才能进行开始和结束段的拟合
            return False
        
        # 使用最小二乘法拟合开始和结束部分的路径段
        num_fit_points = 20  # 用于拟合的点数
        
        # 拟合开始部分的路径段
        start_points = path_points[:num_fit_points]
        start_x = np.array([p[0] for p in start_points])
        start_y = np.array([p[1] for p in start_points])
        start_line = np.polyfit(start_x, start_y, 1)  # 一阶多项式拟合（直线）
        start_slope = start_line[0]  # 斜率
        start_angle = math.atan2(start_slope, 1.0)  # 方向角
        
        # 拟合结束部分的路径段
        end_points = path_points[-num_fit_points:]
        end_x = np.array([p[0] for p in end_points])
        end_y = np.array([p[1] for p in end_points])
        end_line = np.polyfit(end_x, end_y, 1)  # 一阶多项式拟合（直线）
        end_slope = end_line[0]  # 斜率
        end_angle = math.atan2(end_slope, 1.0)  # 方向角
        
        # 计算方向角变化
        angle_diff = abs(end_angle - start_angle)
        # 归一化到 [0, pi]范围内的角度差
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        # 使用所有收集到的点进行圆拟合，计算曲率半径（保持原有代码）
        try:
            # 将路径点转换为numpy数组
            points_array = np.array(path_points)
            x_values = points_array[:, 0]
            y_values = points_array[:, 1]
            
            # 步骤1：计算需要的累加值
            n = len(x_values)
            sum_x = np.sum(x_values)
            sum_y = np.sum(y_values)
            sum_xx = np.sum(x_values*x_values)
            sum_yy = np.sum(y_values*y_values)
            sum_xy = np.sum(x_values*y_values)
            sum_xxx = np.sum(x_values*x_values*x_values)
            sum_xxy = np.sum(x_values*x_values*y_values)
            sum_xyy = np.sum(x_values*y_values*y_values)
            sum_yyy = np.sum(y_values*y_values*y_values)
            
            # 步骤2：构建并求解方程组
            A = np.array([
                [sum_xx, sum_xy, sum_x],
                [sum_xy, sum_yy, sum_y],
                [sum_x, sum_y, n]
            ])
            B = np.array([
                [-(sum_xxx + sum_xyy)],
                [-(sum_xxy + sum_yyy)],
                [-(sum_xx + sum_yy)]
            ])
            
            # 求解圆的参数方程
            X = np.linalg.solve(A, B)
            a, b, c = X.flatten()
            
            # 计算圆心和半径
            center_x = -a/2
            center_y = -b/2
            radius = math.sqrt(center_x*center_x + center_y*center_y - c)
            
            # 曲率半径就是圆的半径
            curvature_radius = radius
            
        except Exception as e:
            # 计算错误，默认曲率半径为无穷大
            self.get_logger().warning(f"圆拟合计算失败: {e}")
            curvature_radius = float('inf')
        
        # 判断是否需要掉头：曲率半径小于阈值或角度变化大于阈值
        # 注意: 这里使用我们新计算的角度差异
        needs_turn_around = (curvature_radius < self.min_curvature_radius and
                            angle_diff > self.min_angle_change)
        
        self.get_logger().debug(f"当前曲率半径: {curvature_radius:.2f}m, 角度变化: {angle_diff:.1f}")
        
        if needs_turn_around:
            self.get_logger().info(f"检测到需要掉头 - 曲率半径: {curvature_radius:.2f}m, 方向角变化: {angle_diff:.1f}, 起始角: {start_angle:.1f}, 结束角: {end_angle:.1f}")
        
        return needs_turn_around

    # 添加掉头控制函数
    def turn_around_control(self):
        """
        执行三点掉头控制
        """
        # 如果红灯亮起且未开始掉头，暂缓掉头
        if not self.turn_around_in_progress and self.traffic_light_state in ['red', 'yellow']:
            self.get_logger().info(f'因红绿灯状态({self.traffic_light_state})而暂缓掉头执行')
            self.stop_vehicle()
            return
        
        current_time = time.time()
        
        # 如果刚开始掉头，初始化状态
        if not self.turn_around_in_progress:
            self.turn_around_in_progress = True
            self.turn_around_start_time = current_time
            self.turn_around_step = 0
            self.turn_around_last_command_time = 0
            self.get_logger().info("开始执行三点掉头")
        
        # 掉头步骤的持续时间（参考turn_around.py）
        step_durations = [
            3.30,   # 步骤1: 直线前行
            0.5,   # 停止
            2.25,  # 步骤2: 向左急转
            0.5,   # 停止
            1.30,  # 步骤3: 右转倒车
            0.5,   # 停止
            1.0,   # 步骤4: 向左急转
            0.5,   # 停止
            1.30,  # 步骤5: 右转倒车
            0.5,   # 停止
            3.7,  # 步骤6: 向左急转
            0.5    # 最后停止
        ]
        
        # 掉头步骤的控制命令（转向角、速度）
        step_commands = [
            (-0.07, self.turn_around_normal_speed),   # 步骤1: 直线前行
            (0.0, 0.0),                             # 停止
            (self.turn_around_max_steering, self.turn_around_slow_speed),  # 步骤2: 向左急转
            (0.0, 0.0),                             # 停止
            (-self.turn_around_max_steering, -self.turn_around_slow_speed), # 步骤3: 右转倒车
            (0.0, 0.0),                             # 停止
            (self.turn_around_max_steering, self.turn_around_slow_speed),  # 步骤4: 向左急转
            (0.0, 0.0),                             # 停止
            (-self.turn_around_max_steering, -self.turn_around_slow_speed), # 步骤5: 右转倒车
            (0.0, 0.0),                             # 停止
            (self.turn_around_max_steering, self.turn_around_slow_speed),  # 步骤6: 向左急转
            (0.0, 0.0)                              # 最后停止
        ]
        
        # 掉头步骤的描述
        step_descriptions = [
            "步骤1: 直线前行",
            "停止",
            "步骤2: 向左急转",
            "停止",
            "步骤3: 右转倒车",
            "停止",
            "步骤4: 向左急转",
            "停止", 
            "步骤5: 右转倒车",
            "停止",
            "步骤6: 向左急转",
            "完成掉头，停止"
        ]
        
        # 计算当前步骤
        current_step = self.turn_around_step
        elapsed_time = current_time - self.turn_around_start_time
        
        # 计算步骤总耗时
        total_elapsed = 0
        for i in range(len(step_durations)):
            total_elapsed += step_durations[i]
            if elapsed_time < total_elapsed:
                current_step = i
                break
        
        # 添加这行：如果已经超过总时间，设置为最后一步之后
        if elapsed_time >= total_elapsed:
            current_step = len(step_commands)
        
        # 如果已完成所有步骤，重置掉头状态
        if current_step >= len(step_commands):
            self.get_logger().info("三点掉头完成")
            self.turn_around_in_progress = False
            
            # 设置一个小的初速度帮助车辆启动
            self.steering_angle = 0.0
            self.motor_throttle = 0.1
            self.send_command()
            
            self.just_completed_turn_around = True
            self.turn_around_completion_time = current_time
            
            # 强制设置下一个导航模式
            if self.prev_nav_mode and self.prev_nav_mode != "turn around":
                self.current_nav_mode = self.prev_nav_mode
            else:
                self.current_nav_mode = "follow path"
                
            self.find_target_point(self.current_pose)
            return
        
        # 如果步骤发生变化，输出日志
        if current_step != self.turn_around_step:
            self.turn_around_step = current_step
            self.get_logger().info(f"掉头 - {step_descriptions[current_step]}")
        
        # 应用当前步骤的控制命令
        steering, throttle = step_commands[current_step]
        self.steering_angle = steering
        self.motor_throttle = throttle
        
        # 发送控制命令
        self.send_command()
    
    def determine_nav_mode(self):
        """确定当前的导航模式"""
        # 检查是否正在执行掉头
        if self.turn_around_in_progress:
            return "turn around"
        
        # 检查是否需要掉头
        if self.check_turn_around_condition(self.current_pose):
            return "turn around"
        
        # 原有的导航模式判断逻辑...
        if not self.current_pixel_pos:
            return "follow path"  # 默认使用路径跟踪
        
        # 获取当前道路ID
        current_road_id = self.find_road_at_pixel(self.current_pixel_pos)
        
        # 如果有路径和前瞻点，检查前瞻点所在的道路
        lookahead_road_id = None
        if self.path and len(self.path.poses) > 0:
            lookahead_point = self.find_target_point(self.current_pose)
            if lookahead_point:
                lookahead_pixel = self.ros_to_pixel(lookahead_point.x, lookahead_point.y)
                if lookahead_pixel:
                    lookahead_road_id = self.find_road_at_pixel(lookahead_pixel)
        
        # 决策逻辑
        effective_road_id = current_road_id
        
        # 如果当前在路上，但前瞻点在不同的路上或路外，优先考虑前瞻点
        if current_road_id is not None and lookahead_road_id != current_road_id:
            self.get_logger().debug(f"道路转换: 当前={current_road_id}, 前瞻={lookahead_road_id}. 使用前瞻点。")
            effective_road_id = lookahead_road_id
        # 如果当前在路外，但前瞻点在路上
        elif current_road_id is None and lookahead_road_id is not None:
            self.get_logger().debug(f"接近道路: 当前=None, 前瞻={lookahead_road_id}. 使用前瞻点。")
            effective_road_id = lookahead_road_id
        
        # 根据effective_road_id决定导航模式
        if effective_road_id is None:
            # 不在任何已知道路上
            return "follow path"
        elif effective_road_id == self.loop_road_id:
            # 在 loop 道路上
            return "follow loop"
        else:
            # 在其他道路上
            return "follow yellow line"
    
    def control_loop(self):
        """主控制循环，计算并发布控制命令"""
        # 计算控制周期
        current_time = time.time()
        dt = current_time - self.last_control_time
        self.last_control_time = current_time
        
        # 获取当前位置
        current_pose = self.get_current_pose()
        if not current_pose:
            self.get_logger().warning('无法获取当前位置')
            self.stop_vehicle()
            return
        
        # 确定当前导航模式
        self.prev_nav_mode = self.current_nav_mode
        self.current_nav_mode = self.determine_nav_mode()
        
        # 如果导航模式发生变化，记录时间
        if self.current_nav_mode != self.prev_nav_mode:
            self.mode_change_time = current_time
            self.get_logger().info(f'导航模式变更: {self.prev_nav_mode} -> {self.current_nav_mode}')
            
            # 检查是否刚刚切换到路径跟踪模式
            if self.current_nav_mode == "follow path":
                self.mode_just_changed_to_path_follow = True
                
                # 检查当前红绿灯状态
                if self.traffic_light_state in ['red', 'yellow']:
                    self.path_follow_blocked = True
                    self.get_logger().info(f'路径跟踪模式启动时红绿灯为{self.traffic_light_state}状态，暂时阻塞行驶')
                else:
                    self.path_follow_blocked = False
            
            # 如果切换到loop模式，重置loop_mode_active标志
            if self.current_nav_mode == "follow loop":
                self.loop_mode_active = False
                self.get_current_pose()  # 更新当前位姿
                self.previous_road_id = self.find_road_at_pixel(self.current_pixel_pos)
                self.get_logger().info(f'从道路ID: {self.previous_road_id} 进入环道')
        else:
            # 如果模式没有变化，重置标志
            self.mode_just_changed_to_path_follow = False
        
        # 发布导航命令
        if self.publish_nav_commands:
            nav_msg = String()
            nav_msg.data = self.current_nav_mode
            self.nav_command_publisher.publish(nav_msg)
        
        # 根据不同的导航模式执行相应的控制逻辑
        if self.current_nav_mode == "follow path":
            self.follow_path_control(current_pose)
        elif self.current_nav_mode == "follow yellow line":
            # 将当前位置传递给黄线跟踪函数，以便在需要时回退到路径跟踪
            self.follow_yellow_line_control(current_pose)
        elif self.current_nav_mode == "follow loop":
            # 检查是否需要延迟启动loop模式
            if not self.loop_mode_active:
                entry_strategy = self.loop_entry_strategies.get(
                    self.previous_road_id, 
                    self.loop_entry_strategies["default"]
                )
                initial_steering, initial_speed, delay_time, _ = entry_strategy
                if current_time - self.mode_change_time >= delay_time:
                    self.loop_mode_active = True
                    self.get_logger().info(f'环道跟踪模式已激活 (来自道路ID: {self.previous_road_id})')
                else:
                    # 延迟期间继续使用路径跟踪
                    # self.follow_path_control(current_pose)
                    self.steering_angle = initial_steering
                    self.motor_throttle = initial_speed
                    self.send_command()
                    self.get_logger().info(f'当前速度: {self.motor_throttle:.2f}, 当前转向角: {self.steering_angle:.2f}')
                    self.get_logger().info(f'环道入口调整中... (来自道路ID: {self.previous_road_id}) 还需: {delay_time - (current_time - self.mode_change_time):.2f}秒')
                    return
            
            self.follow_loop_control()
        elif self.current_nav_mode == "turn around":
            # 执行掉头控制
            self.turn_around_control()
            if self.current_nav_mode == "turn around" and not self.turn_around_in_progress and self.just_completed_turn_around:
                # 刚完成掉头，强制切换到之前的导航模式
                if self.prev_nav_mode and self.prev_nav_mode != "turn around":
                    self.current_nav_mode = self.prev_nav_mode
                else:
                    self.current_nav_mode = "follow path"
    
    def follow_path_control(self, current_pose):
        """纯路径跟踪控制"""
        # 检查路径是否可用
        if not self.path:
            self.get_logger().warning('没有可用的路径')
            self.stop_vehicle()
            return
            
        # 检查是否因红绿灯而阻塞
        if self.path_follow_blocked:
            self.get_logger().info(f'因红绿灯状态({self.traffic_light_state})而阻塞，等待绿灯')
            self.stop_vehicle()
            return
        
        current_pose = self.get_current_pose()
            
        # 找到目标点
        target_point = self.find_target_point(current_pose, new_lookahead_distance=self.LOOKAHEAD_DISTANCE)
        if not target_point:
            self.get_logger().warning('无法找到目标点')
            self.stop_vehicle()
            return
            
        # 计算跟踪误差
        x, y, yaw = current_pose
        dx = target_point.x - x
        dy = target_point.y - y
        
        # 在车体坐标系中的目标方向
        target_yaw = math.atan2(dy, dx)
        
        # 计算转向角
        steering_angle = target_yaw - yaw # - 0.04
        # 归一化到[-pi, pi]
        steering_angle = math.atan2(math.sin(steering_angle), math.cos(steering_angle))
        
        # 计算到目标点的距离
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 纯路径跟踪逻辑
        self.MAX_SPEED = 0.15  # 设置合适的速度
        
        # 限制转向角范围
        steering_angle = max(-self.MAX_STEERING, min(self.MAX_STEERING, steering_angle))
        
        # 计算速度（根据转向角和距离调整）
        motor_throttle = self.MAX_SPEED * (1.0 - abs(steering_angle) / math.pi)
        motor_throttle = max(self.MIN_SPEED, min(self.MAX_SPEED, motor_throttle))
        
        # 平滑控制命令
        self.steering_queue.append(steering_angle)
        self.speed_queue.append(motor_throttle)
        
        smooth_steering = sum(self.steering_queue) / len(self.steering_queue)
        smooth_throttle = sum(self.speed_queue) / len(self.speed_queue)
        
        # 更新当前控制值
        self.steering_angle = smooth_steering
        self.motor_throttle = smooth_throttle
        
        # 检查是否到达终点
        last_point = self.path.poses[-1].pose.position
        dist_to_end = math.sqrt((last_point.x - x)**2 + (last_point.y - y)**2)
        
        if dist_to_end < 0.2:  # 到达终点附近
            self.get_logger().info('到达终点！')
            self.stop_vehicle()
            return
        
        # 发布电机控制命令
        self.send_command()
        
        self.get_logger().debug(f'发送控制命令: 速度={smooth_throttle:.2f}, 转向={smooth_steering:.2f}')
        
        
    def analyze_yellow_line_shape(self, yellow_mask):
        """
        分析黄线形状来判断转向方向
        
        Args:
            yellow_mask: 黄线的二值化掩码
            
        Returns:
            tuple: (is_turning_left, is_turning_right) 转向判断结果
        """
        height, width = yellow_mask.shape
        
        # 收集黄线点
        yellow_points = []
        
        # 采样行，从下往上均匀选择几行进行分析
        sample_rows = np.linspace(height-1, height//3, 10, dtype=int)
        
        for y in sample_rows:
            # 获取当前行的黄线像素索引
            x_indices = np.where(yellow_mask[y, :] > 0)[0]
            if len(x_indices) > 0:
                # 对于黄线，取最右侧点（与车道最接近的点）
                x = np.max(x_indices)
                yellow_points.append((y, x))
        
        # 如果没有足够的点用于拟合，返回默认值
        if len(yellow_points) < 3:
            # self.get_logger().warning("分析黄线形状: 没有足够的黄线点用于拟合")
            return False, False
        
        # 进行多项式拟合
        y_values = np.array([p[0] for p in yellow_points])
        x_values = np.array([p[1] for p in yellow_points])
        
        try:
            # 通过二阶多项式拟合黄线形状
            poly_coeffs = np.polyfit(y_values, x_values, 2)
            
            # 二阶多项式的系数：ax^2 + bx + c，其中a表示曲率
            a, b, c = poly_coeffs
            
            # 斜率变化来判断转向
            slope_change = 0
            
            # 设置判断阈值
            turn_threshold = 0.002
            
            is_turning_left = a < -1 * turn_threshold
            is_turning_right = a > turn_threshold
            
            turn_direction = "左转" if is_turning_left else ("右转" if is_turning_right else "直行")
            self.get_logger().debug(f"黄线形状分析 - a: {a:.4f}, 斜率变化: {slope_change:.4f}, 判断: {turn_direction}")
            
            return is_turning_left, is_turning_right
            
        except Exception as e:
            self.get_logger().error(f"黄线形状拟合失败: {e}")
            return False, False
    
    def follow_yellow_line_control(self, current_pose=None):
        """黄线跟踪控制 - 完全还原yellow_follow.py的参数和逻辑"""
        if self.latest_image is None:
            self.get_logger().warning('黄线跟踪: 无可用图像')
            if current_pose and self.path:
                self.get_logger().debug('黄线跟踪: 无可用图像，回退到路径跟踪')
                self.follow_path_control(current_pose)
            else:
                self.stop_vehicle()
            return
        
        current_pose = self.get_current_pose()
        
        # 检查是否到达终点
        if current_pose and self.path and len(self.path.poses) > 0:
            x, y, _ = current_pose
            last_point = self.path.poses[-1].pose.position
            dist_to_end = math.sqrt((last_point.x - x)**2 + (last_point.y - y)**2)
            
            if dist_to_end < 0.2:  # 到达终点附近
                self.get_logger().info('已到达导航终点，停止黄线跟踪！')
                self.stop_vehicle()
                return
        
        # 裁剪图像
        height, width = self.latest_image.shape[:2]
        cropped_image = self.latest_image[self.crop_h_start:self.crop_h_end, 0:self.crop_w]
        
        if cropped_image.size == 0:
            self.get_logger().warning('裁剪后的图像为空')
            if current_pose and self.path:
                self.get_logger().debug('黄线跟踪: 裁剪后图像为空，回退到路径跟踪')
                self.follow_path_control(current_pose)
            else:
                self.stop_vehicle()
            return
        
        # 转换到HSV色彩空间
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        
        # 二值化阈值处理以提取不同颜色区域
        yellow_mask = cv2.inRange(hsv_image, self.yellow_hsv_lower, self.yellow_hsv_upper)
        black_mask = cv2.inRange(hsv_image, self.black_hsv_lower, self.black_hsv_upper)
        white_mask = cv2.inRange(hsv_image, self.white_hsv_lower, self.white_hsv_upper)
        
        # 分析黄线形状判断转向方向
        is_turning_left, is_turning_right = self.analyze_yellow_line_shape(yellow_mask)
        
        # 应用膨胀操作，使虚线连成实线
        yellow_kernel = np.ones((self.yellow_dilation_size, self.yellow_dilation_size), np.uint8)
        white_kernel = np.ones((self.white_dilation_size, self.white_dilation_size), np.uint8)
        
        # 保存原始掩码用于显示
        yellow_mask_original = yellow_mask.copy()
        white_mask_original = white_mask.copy()
        
        # 应用膨胀操作
        yellow_mask = cv2.dilate(yellow_mask, yellow_kernel, iterations=1)
        white_mask = cv2.dilate(white_mask, white_kernel, iterations=1)
        
        # 调试图像
        debug_img = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 3), dtype=np.uint8)
        debug_img[:, :, 0] = white_mask  # 蓝色通道显示白线
        debug_img[:, :, 1] = yellow_mask  # 绿色通道显示黄线
        debug_img[:, :, 2] = black_mask  # 红色通道显示黑色区域
        
        # 查找车道中心
        lane_center_x, lane_found = self.find_lane_center(yellow_mask, black_mask, white_mask)
        
        if not lane_found:
            self.get_logger().warning('黄线跟踪: 未检测到足够的黄线点')
            if current_pose and self.path:
                # 检查红绿灯状态
                if self.traffic_light_state in ['red', 'yellow']:
                    self.get_logger().info(f'黄线跟踪: 未检测到黄线且检测到{self.traffic_light_state}灯，停止等待')
                    self.stop_vehicle()
                else:
                    self.get_logger().info('黄线跟踪: 未检测到黄线，直线一段距离')
                    # self.follow_path_control(current_pose)
                    self.steering_angle = -0.05
                    self.motor_throttle = 0.15
                    self.send_command()
            else:
                self.get_logger().warning('黄线跟踪: 无法回退到路径跟踪，停止车辆')
                self.stop_vehicle()
            return
        
        # 计算控制命令
        height, width = cropped_image.shape[:2]
        
        # 获取当前视觉模式
        was_dual_line_mode = hasattr(self, '_last_detection_info') and self._last_detection_info == "dual"
        is_dual_line_mode = hasattr(self, '_last_detection_info') and self._last_detection_info == "dual"
        
        # 获取图像宽度
        base_center_x = width // 2
        
        # 设置对应模式的图像中心点偏移
        if is_dual_line_mode:
            # 如果从单线模式恢复到双线模式，退出右转单线固定转向模式
            if self.right_turn_single_mode:
                self.right_turn_single_mode = False
                self.get_logger().debug("恢复双线模式，退出右转固定转向")
            
            # 双线模式
            if is_turning_right:
                # 双线右转
                image_center_x = base_center_x + 0
                self.get_logger().debug("双线右转模式: 中心点偏移 +0")
            else:
                # 双线左转或直行
                image_center_x = base_center_x - 65
                self.get_logger().debug("双线左转/直行模式: 中心点偏移 -65")
            
            # 记录当前双线模式下的转向角，用于可能的模式切换
            if not self.right_turn_single_mode:
                # 只在非固定转向模式下更新，以便在模式切换时有准确的值
                self.last_dual_steering = self.steering_angle
        else:
            # 单线模式 - 检查是否从双线模式切换过来
            if was_dual_line_mode:
                # 从双线模式切换到单线模式
                if is_turning_right:
                    # 右转模式下切换到单线：使用固定转向
                    self.right_turn_single_mode = True
                    self.get_logger().debug(f"从双线右转切换到单线模式：使用固定转向角 {self.last_dual_steering - 0.05:.3f}")
            
            # 根据不同情况处理单线模式
            if self.right_turn_single_mode:
                # 右转单线固定转向模式
                image_center_x = base_center_x  # 这个值不重要，因为我们将使用固定转向
                self.get_logger().debug(f"右转单线固定转向模式: 转向值={self.last_dual_steering - 0.05:.3f}")
            else:
                # 常规单线跟随模式（左转或直行）
                image_center_x = base_center_x
                self.get_logger().debug(f"单线跟随黄线模式: 使用图像中心")
        
        # 根据模式确定控制逻辑
        if self.right_turn_single_mode:
            # 在右转单线固定转向模式下，使用固定转向角
            self.steering_angle = self.last_dual_steering - 0.13
            
            # 计算基于当前转向角的车速
            # 使用右转减速因子
            speed_factor = 1.0 - self.right_reduction * (abs(self.steering_angle) / self.max_steering)**0.7
            self.motor_throttle = self.target_speed * speed_factor
            self.motor_throttle = max(0.05, self.motor_throttle)  # 确保最小速度不要太低
            
            turn_type = "右转固定"
            # 增加连续右转计数
            self.consecutive_right_turns += 1
            self.get_logger().debug(f"连续右转计数: {self.consecutive_right_turns}")
            
            error = 0  # 不重要，仅用于日志
            normalized_error = 0  # 不重要，仅用于日志
        else:
            # 正常的基于误差的控制逻辑
            # 计算基于调整后图像中心的误差
            error = image_center_x - lane_center_x
            
            # 归一化误差（使用调整后的图像中心）
            normalized_error = error / image_center_x
            
            # 计算当前时间和时间差
            current_time = self.get_clock().now()
            dt = (current_time - self.last_time).nanoseconds / 1e9  # 转换为秒
            self.last_time = current_time
            
            # 计算积分项(I)
            self.integral_error += normalized_error * dt
            # 限制积分项大小，防止积分饱和
            self.integral_error = max(-self.max_integral, min(self.max_integral, self.integral_error))
            
            # 根据转向方向选择对应的控制参数
            if is_turning_left:
                # 左转
                current_gain = self.left_gain
                current_reduction = self.left_reduction
                turn_type = "左转"
                # 重置连续右转计数
                self.consecutive_right_turns = 0
            elif is_turning_right:
                # 右转
                current_gain = self.right_gain
                current_reduction = self.right_reduction
                turn_type = "右转"
                # 增加连续右转计数
                self.consecutive_right_turns += 1
                self.get_logger().debug(f"连续右转计数: {self.consecutive_right_turns}")
            else:
                # 直行
                current_gain = self.steering_gain
                current_reduction = self.speed_reduction
                turn_type = "直行"
                # 重置连续右转计数
                self.consecutive_right_turns = 0
            
            # 使用选择的参数计算PI控制器输出
            steering = current_gain * normalized_error
            
            # 当误差很小时，逐渐减小积分项，防止小幅震荡
            if abs(normalized_error) < 0.1:
                self.integral_error *= 0.95  # 缓慢衰减
            
            # 保存当前误差用于下一次计算
            self.last_error = normalized_error
            
            # 平滑转向命令，减少突然变化
            smoothing_factor = 0.7  # 使用yellow_follow.py中的平滑因子
            self.steering_angle = smoothing_factor * self.last_steering + (1 - smoothing_factor) * np.clip(steering, -self.max_steering, self.max_steering)
            self.last_steering = self.steering_angle
            
            # 根据转向角调整速度
            speed_factor = 1.0 - current_reduction * (abs(self.steering_angle) / self.max_steering)**0.7
            self.motor_throttle = self.target_speed * speed_factor
            self.motor_throttle = max(0.05, self.motor_throttle)  # 确保最小速度不要太低
        
        # 应用附加平滑过滤
        self.steering_queue.append(self.steering_angle)
        self.speed_queue.append(self.motor_throttle)
        
        if len(self.steering_queue) == self.filter_size:
            self.steering_angle = sum(self.steering_queue) / self.filter_size
            self.motor_throttle = sum(self.speed_queue) / self.filter_size

        # 在日志中显示当前使用的模式和中心点偏移
        if self.right_turn_single_mode:
            mode_str = "右转单线固定"
        else:
            mode_str = "双线" if is_dual_line_mode else "黄线跟随"
        
        self.get_logger().debug(f'模式:{mode_str} {turn_type}, 中心:{image_center_x}, 找到车道中心: {lane_center_x:.1f}, 误差: {error:.1f}, 转向: {self.steering_angle:.3f}, 油门: {self.motor_throttle:.3f}')
        
        # 发布电机控制命令
        self.send_command()

    def find_lane_center(self, yellow_mask, black_mask, white_mask):
        """计算车道中心位置，并使用多项式拟合车道线"""
        height, width = yellow_mask.shape
        
        # 创建车道掩码图像
        lane_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 采样行索引 - 均匀分布在图像高度范围内
        sample_rows = np.linspace(0, height-1, self.sample_points, dtype=int)
        
        # 收集左侧黄线和右侧白线的采样点
        yellow_points = []  # 格式: (y, x)
        white_points = []   # 格式: (y, x)
        
        # 添加最大距离参数，超过这个距离的白线点会被剔除
        max_white_distance = 500  # 黄线和白线的最大距离阈值
        
        # 采样黄线点（取每行最右侧点）
        for y in sample_rows:
            x_indices = np.where(yellow_mask[y, :] > 0)[0]
            if len(x_indices) > 0:
                yellow_x = np.max(x_indices)  # 取最右侧点
                yellow_points.append((y, yellow_x))
                
                # 改进的白线采样策略：检查黄线右侧是否有足够宽度的黑色区域，然后找白色点
                if yellow_x + 1 < width:  # 确保黄线右侧有空间
                    # 计算黄线右侧的搜索区域上限
                    search_limit = min(width, yellow_x + 200)  # 防止越界，最多搜索200像素
                    
                    # 检查黄线右侧是否有黑色区域
                    # 先找到黄线右侧的所有黑色像素
                    black_pixels = np.where(black_mask[y, yellow_x:search_limit] > 0)[0]
                    
                    if len(black_pixels) > 0:
                        # 检查黑色区域是否足够宽（至少75像素）或者延伸到搜索区域边缘
                        black_width = np.max(black_pixels) - np.min(black_pixels) + 1
                        if black_width >= 75 or np.max(black_pixels) >= (search_limit - yellow_x - 1):
                            # 确定开始寻找白线的位置
                            # 我们从黄线右侧至少75像素处开始寻找白线
                            white_search_start = yellow_x + max(75, np.max(black_pixels) + 1)
                            
                            if white_search_start < width:
                                # 在适当位置寻找白线点
                                white_x_indices = np.where(white_mask[y, white_search_start:] > 0)[0]
                                
                                if len(white_x_indices) > 0:
                                    # 最左侧的白线点（加上搜索起始位置的偏移）
                                    white_x = white_search_start + white_x_indices[0]
                                    
                                    # 计算白线点与黄线点的距离
                                    distance = white_x - yellow_x
                                    
                                    # 只有距离在合理范围内的白线点才被采用
                                    if distance <= max_white_distance:
                                        white_points.append((y, white_x))
        
        # 默认值
        lane_center_x = width // 2
        lane_found = False
        
        # 多项式拟合黄线
        yellow_poly = None
        if len(yellow_points) >= self.min_points:
            y_yellow = np.array([p[0] for p in yellow_points])
            x_yellow = np.array([p[1] for p in yellow_points])
            yellow_poly = np.polyfit(y_yellow, x_yellow, self.poly_order)
        
        # 多项式拟合白线
        white_poly = None
        if len(white_points) >= self.min_points:
            y_white = np.array([p[0] for p in white_points])
            x_white = np.array([p[1] for p in white_points])
            white_poly = np.polyfit(y_white, x_white, self.poly_order)
        
        # 检查是否成功拟合车道线
        if yellow_poly is not None:
            # 情况1: 同时拟合到黄线和白线
            if white_poly is not None:
                lane_found = True
                
                # 生成车道区域掩码
                lane_area_points = []
                lane_widths = []  # 存储每行的车道宽度
                for y in range(height):
                    x_left = np.polyval(yellow_poly, y)
                    x_right = np.polyval(white_poly, y)
                    
                    if 0 <= x_left < width and 0 <= x_right < width and x_left < x_right:
                        # 计算每一行的车道宽度
                        current_width = x_right - x_left
                        lane_widths.append(current_width)
                        
                        # 添加车道区域的左右边界点
                        lane_area_points.append([(int(x_left), y), (int(x_right), y)])
                        
                        # 填充车道区域掩码
                        cv2.line(lane_mask, (int(x_left), y), (int(x_right), y), 255, 1)
                
                # 更新测量的车道宽度（使用下半部分的平均值和最大值）
                if lane_widths:
                    # 取下半部分的车道宽度
                    lower_half_widths = lane_widths[len(lane_widths)//2:]
                    if lower_half_widths:
                        # 计算平均车道宽度
                        self.last_measured_lane_width = np.mean(lower_half_widths)
                        # 更新历史最大车道宽度
                        current_max_width = np.max(lower_half_widths)
                        if current_max_width > self.max_measured_lane_width:
                            self.max_measured_lane_width = current_max_width
                            self.get_logger().debug(f"更新最大车道宽度: {self.max_measured_lane_width:.1f}像素")
                        self.get_logger().debug(f"当前车道宽度: {self.last_measured_lane_width:.1f}像素")

                # 计算整体车道中心位置 (使用下半部分的中心点平均值)
                center_points = []
                for y in range(height // 2, height):
                    x_left = np.polyval(yellow_poly, y)
                    x_right = np.polyval(white_poly, y)
                    if 0 <= x_left < width and 0 <= x_right < width and x_left < x_right:
                        center_points.append((x_left + x_right) / 2)
                
                if center_points:
                    lane_center_x = np.mean(center_points)
                    self.get_logger().debug(f"双线拟合模式：中心点={lane_center_x:.1f}")
                self._last_detection_info = "dual"  # 记录当前是双线模式
                
            # 情况2: 只拟合到黄线 - 改为直接基于黄线位置进行循迹
            else:
                lane_found = True
                # 直接根据黄线位置加上固定偏移量作为参考点
                center_points = []
                for y in range(height // 2, height):
                    x_left = np.polyval(yellow_poly, y)
                    if 0 <= x_left < width:
                        # 使用黄线位置加上固定偏移量作为跟踪参考点
                        x_reference = x_left + self.yellow_line_offset
                        center_points.append(x_reference)
                        
                        # 为掩码添加参考区域
                        cv2.line(lane_mask, (int(x_left), y), (int(x_reference), y), 255, 1)
                
                if center_points:
                    lane_center_x = np.mean(center_points)
                    self.get_logger().debug(f"单线循迹模式：直接跟随黄线，参考点={lane_center_x:.1f}，偏移量={self.yellow_line_offset}")
                self._last_detection_info = "single"  # 记录当前是单线模式
        else:
            # 未检测到足够的黄线点进行拟合
            self.get_logger().warn("未检测到足够的黄线点进行拟合，车辆将停止")
            lane_found = False
            self._last_detection_info = "none"  # 未检测到车道线
        
        # 限制在图像范围内（如果找到了车道）
        if lane_found:
            lane_center_x = max(0, min(width-1, lane_center_x))
        
        return lane_center_x, lane_found
    
    def follow_loop_control(self):
        """环道跟踪控制 - 完全还原loop_follow.py的参数和逻辑"""
        if self.latest_image is None:
            self.get_logger().warning('环道跟踪: 无可用图像')
            self.stop_vehicle()
            return
        
        # 检查是否到达终点
        current_pose = self.get_current_pose()
        if current_pose and self.path and len(self.path.poses) > 0:
            x, y, _ = current_pose
            last_point = self.path.poses[-1].pose.position
            dist_to_end = math.sqrt((last_point.x - x)**2 + (last_point.y - y)**2)
            
            if dist_to_end < 0.2:  # 到达终点附近
                self.get_logger().info('已到达导航终点，停止环道跟踪！')
                self.stop_vehicle()
                return
        
        # 裁剪图像
        height, width = self.latest_image.shape[:2]
        cropped_image = self.latest_image[self.loop_crop_h_start:self.loop_crop_h_end, 0:self.loop_crop_w]
        
        if cropped_image.size == 0:
            self.get_logger().warning('裁剪后的图像为空')
            if current_pose and self.path:
                self.get_logger().info('环道跟踪: 裁剪后图像为空，回退到路径跟踪')
                self.follow_path_control(current_pose)
            else:
                self.stop_vehicle()
            return
        
        # 转换到HSV色彩空间
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        
        # 提取黑色和白色区域
        black_mask = cv2.inRange(hsv_image, self.loop_black_lower, self.loop_black_upper)
        white_mask = cv2.inRange(hsv_image, self.loop_white_lower, self.loop_white_upper)
        
        # 白色掩码取反得到黑色掩码
        white_mask_inv = cv2.bitwise_not(white_mask)
        
        # 执行膨胀操作
        white_kernel = np.ones((self.loop_dilation_size, self.loop_dilation_size), np.uint8)
        white_mask_dilated = cv2.dilate(white_mask, white_kernel, iterations=1)
        
        # 膨胀后的白色掩码再次取反得到新的黑色掩码
        black_mask_processed = cv2.bitwise_not(white_mask_dilated)
        
        # 创建调试图像
        debug_img = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 3), dtype=np.uint8)
        debug_img[:, :, 0] = white_mask  # 蓝色通道显示白线
        debug_img[:, :, 2] = black_mask  # 红色通道显示黑色区域
        
        # 查找左右白线
        height, width = black_mask_processed.shape
        left_white_points = []
        right_white_points = []
        sample_rows = np.linspace(0, height-1, self.loop_sample_points, dtype=int)
        
        for y in sample_rows:
            # 获取当前行的黑色像素索引
            black_indices = np.where(black_mask[y, :] > 0)[0]
            
            # 如果该行没有足够的黑色像素，跳过
            if len(black_indices) < self.loop_black_divider_points:
                continue
            
            # 从左向右数，找到第N个黑色像素的位置
            divider_x = black_indices[self.loop_black_divider_points - 1]
            
            # 在分界线左侧寻找左侧白线（取最右侧白点）
            left_indices = np.where(white_mask[y, :divider_x] > 0)[0]
            if len(left_indices) > 0:
                left_x = np.max(left_indices)  # 最右侧的白点作为左白线
                left_white_points.append((y, left_x))
                cv2.circle(debug_img, (int(left_x), int(y)), 3, (255, 0, 0), -1)  # 蓝色标记左白线点
            
            # 在分界线右侧寻找右侧白线（取最左侧白点）
            right_indices = np.where(white_mask[y, divider_x:] > 0)[0]
            if len(right_indices) > 0:
                right_x = divider_x + np.min(right_indices)  # 最左侧的白点作为右白线
                right_white_points.append((y, right_x))
                cv2.circle(debug_img, (int(right_x), int(y)), 3, (0, 0, 255), -1)  # 红色标记右白线点
                
                # 可视化黑色分界线
                cv2.line(debug_img, (int(divider_x), int(y)), (int(divider_x), int(y)), (0, 255, 0), 3)  # 绿色点标记分界线
        
        # 检查是否找到足够的白线点
        lane_found = False
        
        if len(right_white_points) >= self.loop_min_points:
            # 只有右侧白线
            y_right = np.array([p[0] for p in right_white_points])
            x_right = np.array([p[1] for p in right_white_points])
            right_poly = np.polyfit(y_right, x_right, self.loop_poly_order)
            
            # 绘制拟合右线
            for y in range(height):
                x_right = np.polyval(right_poly, y)
                if 0 <= x_right < width:
                    cv2.circle(debug_img, (int(x_right), y), 1, (0, 255, 255), -1)  # 黄红色标记右线拟合曲线
            
            # 根据右侧白线位置和估计车道宽度计算车道中心
            center_points = []
            for y in range(height // 2, height):
                x_right = np.polyval(right_poly, y)
                if 0 <= x_right < width:
                    # 估计车道中心位置
                    x_center = x_right - self.loop_lane_width / 2
                    if x_center > 0:
                        center_points.append(x_center)
                        cv2.circle(debug_img, (int(x_center), y), 1, (255, 255, 255), -1)  # 白色标记估计的车道中心
            
            if center_points:
                lane_found = True
                lane_center_x = np.mean(center_points)
                # 使用偏移的图像中心点
                image_center_x = width // 2 + self.loop_image_center_offset
                
                # 计算误差和控制命令，类似于双线情况
                error = image_center_x - lane_center_x
                normalized_error = error / image_center_x
                
                # 只使用P控制器，简化单线情况
                steering = self.loop_steering_gain * normalized_error
                steering = np.clip(steering, -self.loop_max_steering, self.loop_max_steering)
                
                # 平滑转向命令
                smoothing_factor = 0.7
                steering = smoothing_factor * self.loop_last_steering + (1 - smoothing_factor) * steering
                self.loop_last_steering = steering
                
                # 根据转向角调整速度
                speed_factor = 1.0 - self.loop_speed_reduction * (abs(steering) / self.loop_max_steering)**0.7
                motor_throttle = self.loop_target_speed * speed_factor
                motor_throttle = max(0.05, motor_throttle)  # 确保最小速度不要太低
                
                # 更新当前控制值
                self.steering_angle = steering
                self.motor_throttle = motor_throttle
                
                # 发布电机控制命令
                self.send_command()
                
                self.get_logger().debug(f'环道跟踪(右线): 找到中心={lane_center_x:.1f}, 目标={image_center_x}, 误差={error:.1f}, 转向={steering:.3f}, 油门={motor_throttle:.3f}')
        
        # 显示调试图像
        # cv2.imshow("Loop Detection", debug_img)
        # cv2.waitKey(1)
        
        # 如果没有找到足够的白线
        if not lane_found:
            self.get_logger().warning('环道跟踪: 未检测到足够的白线点')
            
            # 检查是否可以回退到路径跟踪
            if current_pose and self.path:
                self.get_logger().info('环道跟踪: 回退到路径跟踪')
                self.follow_path_control(current_pose)
            else:
                self.get_logger().warning('环道跟踪: 无法回退到路径跟踪，停止车辆')
                self.stop_vehicle()
    
    def send_command(self):
        """发送转向和速度命令到电机"""
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [self.steering_angle, self.motor_throttle]
        self.motor_publisher.publish(msg)
    
    def stop_vehicle(self):
        """停止车辆"""
        self.steering_angle = 0.0
        self.motor_throttle = 0.0
        self.send_command()

def main(args=None):
    """ROS主函数"""
    rclpy.init(args=args)
    
    # 创建路径跟随控制器节点
    controller = PathFollowController()
    
    # 启动节点并保持运行
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        # 停止车辆
        controller.stop_vehicle()
        # 关闭节点
        controller.destroy_node()
        rclpy.shutdown()
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows() 

if __name__ == '__main__':
    main()
