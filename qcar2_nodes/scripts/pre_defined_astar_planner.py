import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion
import math
import os
import pickle
import yaml
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from tf2_ros import TransformException # type: ignore
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import threading
import time
from std_msgs.msg import Header

# 添加角度调整常量 (逆时针旋转48度，转换为弧度)
ANGLE_ADJUSTMENT = - 48.0 * math.pi / 180.0

def parse_map(map_image):
    """解析地图图像，识别道路、黄线、斑马线等元素 (使用BGR颜色空间)"""
    # 确保图像是BGR格式(OpenCV默认)
    if len(map_image.shape) == 3 and map_image.shape[2] == 3:
        img_bgr = map_image.copy()
    else:
        img_bgr = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
    
    # 在BGR空间中定义颜色范围
    # 黑色范围 (道路)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([120, 120, 120])
    black_mask = cv2.inRange(img_bgr, lower_black, upper_black)
    
    # 黄色范围 (黄线)
    lower_yellow = np.array([0, 140, 140])  # BGR格式，对应黄色
    upper_yellow = np.array([100, 255, 255])  # BGR格式，对应黄色
    yellow_mask = cv2.inRange(img_bgr, lower_yellow, upper_yellow)
        
    # 品蓝色范围 (斑马线)
    lower_blue = np.array([190, 50, 0])  # BGR格式，对应蓝色
    upper_blue = np.array([255, 150, 100])  # BGR格式，对应蓝色
    blue_mask = cv2.inRange(img_bgr, lower_blue, upper_blue)
    
    yellow = yellow_mask > 0
    zebra_crossings = blue_mask > 0
    # 可通行区域为黑色区域和斑马线区域
    traversable = black_mask > 0
    traversable = np.logical_or(traversable, zebra_crossings)
    
    # 创建成本地图
    cost_map = np.zeros_like(traversable, dtype=float)
    
    # 斑马线区域增加成本（需要停车等待）
    cost_map[blue_mask > 0] += 1
    
    # 黄线区域设置为高成本（禁止跨越）
    cost_map[yellow_mask > 0] += 1000
    
    return traversable, yellow_mask, blue_mask, cost_map

def is_crossing_yellow_line(pos1, pos2, yellow_line):
    """判断从pos1到pos2的移动是否跨越黄线"""
    y1, x1 = pos1
    y2, x2 = pos2
    
    # 如果是相邻像素，直接检查是否有黄线
    if abs(y2 - y1) <= 1 and abs(x2 - x1) <= 1:
        min_y, max_y = min(y1, y2), max(y1, y2)
        min_x, max_x = min(x1, x2), max(x1, x2)
        
        # 检查路径上是否有黄线
        if np.any(yellow_line[min_y:max_y+1, min_x:max_x+1]):
            return True
    else:
        # 对于非相邻像素，插值检查
        points = np.linspace(start=(y1, x1), stop=(y2, x2), num=10).astype(int)
        for p in points:
            if yellow_line[p[0], p[1]]:
                return True
    
    return False

def a_star_with_reference_paths(start, goal, traversable, cost_map, yellow_line, reference_paths):
    """带参考路径的A*算法，强制沿参考路径行走且遵循方向"""
    rows, cols = traversable.shape
    
    # 创建参考路径地图 - 只有参考路径附近区域可通行
    reference_path_map = np.zeros((rows, cols), dtype=bool)
    # 创建方向图 - 存储每个点的方向向量
    direction_map = np.zeros((rows, cols, 2), dtype=float)  # [dy, dx]格式
    
    # 将参考路径点及其附近区域标记为可通行，并记录方向
    path_width = 2  # 参考路径的宽度
    
    # 标记所有参考路径点
    for path in reference_paths:
        for i in range(len(path)-1):
            y1, x1 = path[i]
            y2, x2 = path[i+1]
            
            # 计算方向向量
            dy = y2 - y1
            dx = x2 - x1
            length = np.sqrt(dy**2 + dx**2)
            if length > 0:
                norm_dy = dy / length
                norm_dx = dx / length
            else:
                continue
                
            # 在两点之间插值生成密集的点序列
            points = np.linspace(start=(y1, x1), stop=(y2, x2), num=max(int(length*3), 2)).astype(int)
            
            # 标记这些点及其周围区域
            for p in points:
                py, px = p
                if 0 <= py < rows and 0 <= px < cols:
                    # 记录此处的方向
                    direction_map[py, px] = [norm_dy, norm_dx]
                    
                    # 标记点周围的圆形区域
                    for oy in range(-path_width, path_width+1):
                        for ox in range(-path_width, path_width+1):
                            ny, nx = py + oy, px + ox
                            if (0 <= ny < rows and 0 <= nx < cols and 
                                oy*oy + ox*ox <= path_width*path_width):  # 圆形区域
                                reference_path_map[ny, nx] = True
                                
                                # 扩散方向信息
                                if direction_map[ny, nx, 0] == 0 and direction_map[ny, nx, 1] == 0:
                                    direction_map[ny, nx] = [norm_dy, norm_dx]
    
    # 创建一个新的成本地图，非参考路径区域成本极高
    strict_cost_map = cost_map.copy()
    non_path_cost = 1000.0  # 非参考路径区域的高成本
    strict_cost_map[~reference_path_map] += non_path_cost
    
    # 确保起点和终点附近可通行
    start_goal_radius = 10
    for point in [start, goal]:
        y, x = point
        for oy in range(-start_goal_radius, start_goal_radius+1):
            for ox in range(-start_goal_radius, start_goal_radius+1):
                ny, nx = y + oy, x + ox
                if (0 <= ny < rows and 0 <= nx < cols and 
                    traversable[ny, nx] and oy*oy + ox*ox <= start_goal_radius*start_goal_radius):
                    strict_cost_map[ny, nx] = cost_map[ny, nx]  # 恢复原始成本
    
    # 可视化参考路径地图和方向
    ref_path_visual = np.zeros((rows, cols, 3), dtype=np.uint8)
    ref_path_visual[reference_path_map] = [0, 255, 0]  # 绿色表示参考路径
    
    # 用箭头显示方向
    arrow_step = 15  # 每隔15个像素绘制一个箭头
    for y in range(0, rows, arrow_step):
        for x in range(0, cols, arrow_step):
            if reference_path_map[y, x]:
                dy, dx = direction_map[y, x]
                if dy != 0 or dx != 0:  # 如果有方向
                    end_y = int(y + dy * 10)
                    end_x = int(x + dx * 10)
                    if 0 <= end_y < rows and 0 <= end_x < cols:
                        cv2.arrowedLine(ref_path_visual, (x, y), (end_x, end_y), (255, 0, 0), 1)
    
    # 标记起点和终点
    ref_path_visual[start[0], start[1]] = [0, 0, 255]  # 红色表示起点
    ref_path_visual[goal[0], goal[1]] = [255, 255, 0]  # 青色表示终点
    
    # cv2.imshow("Reference Paths and Directions", ref_path_visual)
    # cv2.waitKey(1)
    
    # A*算法实现
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    costs = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    open_set_hash = {start}
    closed_set = set()
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        open_set_hash.remove(current)
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for i, (dy, dx) in enumerate(directions):
            neighbor = (current[0] + dy, current[1] + dx)
            
            # 边界和可通行检查
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols) or not traversable[neighbor]:
                continue
            
            if neighbor in closed_set:
                continue
                
            # 基础移动成本
            move_cost = costs[i] + strict_cost_map[neighbor]
            
            # 检查是否跨越黄线
            if is_crossing_yellow_line(current, neighbor, yellow_line):
                move_cost += 1000
                
            # 检查移动方向与参考路径方向的一致性
            if reference_path_map[neighbor]:
                # 获取当前移动方向
                move_dy, move_dx = dy, dx
                move_len = np.sqrt(move_dy**2 + move_dx**2)
                if move_len > 0:
                    norm_move_dy = move_dy / move_len
                    norm_move_dx = move_dx / move_len
                    
                    # 获取此位置的参考路径方向
                    path_dy, path_dx = direction_map[neighbor]
                    path_len = np.sqrt(path_dy**2 + path_dx**2)
                    
                    if path_len > 0:  # 有方向信息
                        # 计算方向向量的点积，判断是否顺逆行
                        dot_product = norm_move_dy * path_dy + norm_move_dx * path_dx
                        
                        if dot_product > 0.7:  # 方向基本一致(顺行)
                            move_cost -= 0.5  # 奖励顺行
                        elif dot_product < -0.3:  # 方向相反(逆行)
                            move_cost += 10000  # 严厉惩罚逆行
                        elif -0.3 <= dot_product <= 0.7:  # 方向有偏差(横穿)
                            move_cost += 20  # 轻微惩罚横穿
            
            tentative_g_score = g_score.get(current, float('inf')) + move_cost
            
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score_val = tentative_g_score + heuristic(neighbor, goal)
                f_score[neighbor] = f_score_val
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score_val, neighbor))
                    open_set_hash.add(neighbor)
    
    return None

def get_user_pose(window_name, image, message):
    """让用户通过点击和拖动来选择位置和朝向"""
    pose_params = {"start_pos": None, "end_pos": None, "complete": False, "drag_image": None}
    
    def get_pose(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 第一次点击，设置起始位置
            param["start_pos"] = (y, x)
            param["end_pos"] = None
            param["complete"] = False
            print(f"选择{message}位置: ({y}, {x})")
            
        elif event == cv2.EVENT_MOUSEMOVE and param["start_pos"] is not None and not param["complete"]:
            # 鼠标移动，绘制箭头指示方向
            temp_img = image.copy()
            start_y, start_x = param["start_pos"]
            cv2.arrowedLine(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2, tipLength=0.3)
            param["drag_image"] = temp_img
            cv2.imshow(window_name, temp_img)
            
        elif event == cv2.EVENT_LBUTTONUP and param["start_pos"] is not None:
            # 鼠标释放，完成位姿选择
            param["end_pos"] = (y, x)
            param["complete"] = True
            temp_img = image.copy()
            start_y, start_x = param["start_pos"]
            cv2.arrowedLine(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2, tipLength=0.3)
            param["drag_image"] = temp_img
            cv2.imshow(window_name, temp_img)
            
            # 计算朝向角度（弧度）
            dx = x - start_x
            dy = y - start_y
            angle = math.atan2(dy, dx)
            # 应用角度调整
            adjusted_angle = angle + ANGLE_ADJUSTMENT
            print(f"{message}朝向: {math.degrees(angle):.1f}度 -> 调整后: {math.degrees(adjusted_angle):.1f}度")
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, get_pose, pose_params)
    
    # 显示地图
    cv2.imshow(window_name, image)
    print(f"请在图像上点击并拖动来设置{message}的位置和朝向，然后按任意键继续")
    
    while not pose_params["complete"]:
        key = cv2.waitKey(10)
        if key != -1:  # 如果用户按键，则退出
            break
    
    cv2.waitKey(0)  # 等待用户确认位姿选择
    cv2.destroyAllWindows()
    
    # 返回位置和角度
    if pose_params["start_pos"] and pose_params["end_pos"]:
        start_y, start_x = pose_params["start_pos"]
        end_y, end_x = pose_params["end_pos"]
        angle = math.atan2(end_y - start_y, end_x - start_x)
        # 应用角度调整
        adjusted_angle = angle + ANGLE_ADJUSTMENT
        return {"position": pose_params["start_pos"], "angle": adjusted_angle}
    return None

def get_user_click_position(window_name, image, message):
    """让用户通过点击图像来选择位置"""
    click_params = {"position": None}
    
    def get_click_position(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param["position"] = (y, x)  # 注意：OpenCV坐标是(x,y)，但我们内部用(y,x)
            print(f"{message}: ({y}, {x})")
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, get_click_position, click_params)
    
    # 显示地图
    cv2.imshow(window_name, image)
    print(f"请在图像上点击选择{message}，然后按任意键继续")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return click_params["position"]

def visualize_path(map_image, path, vehicle_pose=None):
    """可视化路径规划结果"""
    # 复制原图像以便在其上绘制
    result = map_image.copy()
    
    # 在原图上用红色绘制路径
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i+1]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 标记起点（车辆位置）和终点
    if vehicle_pose:
        y, x = vehicle_pose["position"]
        angle = vehicle_pose["angle"]  # 此角度已经包含调整
        # 绘制车辆位置点
        cv2.circle(result, (x, y), 5, (0, 255, 0), -1)  # 绿色圆形表示起点
        
        # 绘制车辆朝向
        # arrow_length = 20
        # end_x = int(x + arrow_length * math.cos(angle))
        # end_y = int(y + arrow_length * math.sin(angle))
        # cv2.arrowedLine(result, (x, y), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)
    
    if path:
        y, x = path[-1]
        cv2.circle(result, (x, y), 5, (255, 0, 0), -1)  # 蓝色圆形表示终点
    
    return result

def load_transformation_matrix(file_path="map_transform.yaml"):
    """从YAML文件加载转换矩阵"""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        matrix = np.array(data['transformation_matrix'])
        print("已加载坐标转换矩阵")
        return matrix
    except Exception as e:
        print(f"加载转换矩阵失败: {e}")
        return None

def ros_to_pixel_coordinates(ros_x, ros_y, transformation_matrix):
    """将ROS坐标转换为地图像素坐标"""
    # 计算变换矩阵的逆矩阵（从ROS坐标到像素坐标）
    inv_matrix = np.linalg.inv(transformation_matrix)
    
    # 将ROS坐标转换为像素坐标
    ros_point = np.array([[ros_x, ros_y]], dtype=np.float32).reshape(-1, 1, 2)
    pixel_point = cv2.perspectiveTransform(ros_point, inv_matrix)
    
    # 从OpenCV坐标(x,y)转换为我们内部使用的(y,x)格式
    x, y = pixel_point[0][0]
    return (int(round(y)), int(round(x)))

def read_locations_from_file(file_path):
    """从文件中读取QLabs坐标"""
    locations = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # 跳过注释行和空行
                if line.strip() and not line.strip().startswith('//'):
                    # 分割坐标值
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
                        locations.append((x, y))
    except Exception as e:
        print(f"读取位置文件出错: {e}")
    return locations

def load_qlabs_transformation_matrix(file_path="map_transform_qlabs.yaml"):
    """从YAML文件加载QLabs坐标转换矩阵"""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        matrix = np.array(data['transformation_matrix'])
        print("已加载QLabs坐标转换矩阵")
        return matrix
    except Exception as e:
        print(f"加载QLabs转换矩阵失败: {e}")
        return None

def qlabs_to_pixel_coordinates(qlabs_x, qlabs_y, transformation_matrix):
    """将QLabs坐标转换为地图像素坐标"""
    # 计算变换矩阵的逆矩阵（从QLabs坐标到像素坐标）
    inv_matrix = np.linalg.inv(transformation_matrix)
    
    # 将QLabs坐标转换为像素坐标
    qlabs_point = np.array([[qlabs_x, qlabs_y]], dtype=np.float32).reshape(-1, 1, 2)
    pixel_point = cv2.perspectiveTransform(qlabs_point, inv_matrix)
    
    # 从OpenCV坐标(x,y)转换为我们内部使用的(y,x)格式
    x, y = pixel_point[0][0]
    return (int(round(y)), int(round(x)))

class PathPlannerNode(Node):
    """ROS节点：路径规划器"""
    def __init__(self):
        super().__init__('path_planner')
        
        # 创建路径发布器
        self.path_publisher = self.create_publisher(Path, '/planned_path', 10)
        
        # 创建TF监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 加载地图和参考路径
        self.map_image = cv2.imread("ori_map.png")
        self.fmap_image = cv2.imread("final_map.png")
        
        if self.map_image is None or self.fmap_image is None:
            self.get_logger().error("无法读取地图图像")
            return
            
        # 解析地图
        self.traversable, self.yellow_line, self.zebra_crossings, self.cost_map = parse_map(self.fmap_image)
        
        # 加载参考路径
        self.reference_paths = None
        try:
            if os.path.exists('reference_paths.pkl'):
                with open('reference_paths.pkl', 'rb') as f:
                    self.reference_paths = pickle.load(f)
                self.get_logger().info(f"已加载 {len(self.reference_paths)} 条参考路径")
            else:
                self.get_logger().error("未找到参考路径文件 reference_paths.pkl")
        except Exception as e:
            self.get_logger().error(f"加载参考路径失败: {e}")
        
        # 加载坐标转换矩阵
        self.transformation_matrix = load_transformation_matrix()
        if self.transformation_matrix is None:
            self.get_logger().warning("无法加载坐标转换矩阵")
            
        self.qlabs_transformation_matrix = load_qlabs_transformation_matrix("map_transform_qlabs.yaml")
        if self.qlabs_transformation_matrix is None:
            self.get_logger().warning("无法加载QLabs坐标转换矩阵")
        
        # 当前已规划路径
        self.current_path = None
        
        # 创建一个窗口用于接收键盘输入
        self.create_control_window()
        
        # 启动交互规划模式
        self.get_logger().info("路径规划节点已启动，按 'p' 进行路径规划，按 'q' 退出")
        
        # 创建定时器以定期检查键盘输入
        self.create_timer(0.1, self.check_keyboard_input)
        
        # 添加一个标志来防止重复关闭
        self.is_shutting_down = False
    
    def create_control_window(self):
        """创建一个用于接收键盘输入的可缩放控制窗口"""
        # 创建一个小型控制窗口
        control_img = np.ones((300, 500, 3), dtype=np.uint8) * 240  # 浅灰色背景
        
        # 在窗口上显示使用说明
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(control_img, "QCar2 Path Planner", (10, 30), font, 0.8, (0, 0, 0), 2)
        cv2.putText(control_img, "Press 'p' to execute auto 3-stage planning", (10, 70), font, 0.6, (0, 0, 0), 1)
        cv2.putText(control_img, "Press 'q' to quit", (10, 100), font, 0.6, (0, 0, 0), 1)
        # cv2.putText(control_img, "- This window is resizable -", (10, 140), font, 0.6, (0, 0, 150), 1)
        # cv2.putText(control_img, "Drag window edges to resize", (10, 170), font, 0.5, (100, 100, 100), 1)
        
        # 在底部添加装饰性边框
        cv2.rectangle(control_img, (0, 250), (500, 300), (200, 200, 200), -1)
        cv2.putText(control_img, "QCar2 Navigation System", (120, 280), font, 0.6, (80, 80, 80), 1)
        
        # 创建可调整大小的窗口
        cv2.namedWindow("QCar2 Path Planner Control", cv2.WINDOW_NORMAL)
        
        # 设置初始窗口大小
        cv2.resizeWindow("QCar2 Path Planner Control", 500, 425)
        
        # 显示窗口
        cv2.imshow("QCar2 Path Planner Control", control_img)
        cv2.waitKey(1)  # 更新显示
        
        self.get_logger().info("控制窗口已创建，可通过拖动窗口边缘调整大小")

    def check_keyboard_input(self):
        """检查键盘输入以触发规划"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('p'):
            self.get_logger().info("检测到 'p' 键，开始路径规划")
            self.plan_and_publish_path()
        elif key == ord('q'):
            if not self.is_shutting_down:
                self.is_shutting_down = True
                self.get_logger().info("退出路径规划节点")
                # 使用单独的线程来关闭节点，避免在回调中直接调用shutdown
                threading.Thread(target=self.shutdown).start()
    
    def shutdown(self):
        """安全关闭节点"""
        cv2.destroyAllWindows()  # 先关闭所有窗口
        time.sleep(0.5)  # 给OpenCV窗口一些时间关闭
        rclpy.shutdown()
        
    def wait_for_goal_reached(self, goal_point, distance_threshold=0.2):
        """
        等待车辆到达目标点
        
        参数:
            goal_point: 目标点像素坐标 (y, x)
            distance_threshold: 认为已到达的距离阈值（ROS坐标系下，单位：米）
        
        返回:
            bool: 如果成功到达返回True
        """
        # 先将像素坐标的目标点转换为ROS坐标系
        if self.transformation_matrix is not None:
            # 从(y,x)转换为(x,y)格式
            y, x = goal_point
            pixel_point = np.array([[[x, y]]], dtype=np.float32)
            # 使用变换矩阵将像素坐标转换为ROS坐标
            ros_point = cv2.perspectiveTransform(pixel_point, self.transformation_matrix)
            goal_ros_x, goal_ros_y = ros_point[0][0]
            
            self.get_logger().info(f"等待车辆到达目标点 (ROS坐标: {goal_ros_x:.3f}, {goal_ros_y:.3f})，距离阈值: {distance_threshold}米")
            
            # 记录上次日志时间
            last_log_time = time.time()
            check_interval = 0.1  # 检查间隔（秒）
            
            while True:
                current_time = time.time()
                
                try:
                    # 查询从map到base_scan的变换
                    trans = self.tf_buffer.lookup_transform(
                        'map',
                        'base_scan',
                        Time())
                    
                    # 提取当前位置
                    current_ros_x = trans.transform.translation.x
                    current_ros_y = trans.transform.translation.y
                    
                    # 计算与目标点的距离（ROS坐标系下，单位：米）
                    distance = math.sqrt((current_ros_x - goal_ros_x)**2 + (current_ros_y - goal_ros_y)**2)
                    
                    # 每秒记录一次日志
                    if current_time - last_log_time >= 0.1:
                        self.get_logger().info(f"当前距离目标点: {distance:.3f} 米")
                        last_log_time = current_time
                    
                    # 检查是否到达目标点
                    if distance <= distance_threshold:
                        self.get_logger().info(f"已到达目标点！最终距离: {distance:.3f} 米")
                        return True
                    
                except TransformException as ex:
                    if current_time - last_log_time >= 1.0:
                        self.get_logger().warning(f'无法获取车辆位置: {ex}')
                        last_log_time = current_time
                
                # 等待一段时间再检查
                time.sleep(check_interval)
                
                # 处理ROS事件（使定时器等回调能够执行）
                rclpy.spin_once(self, timeout_sec=0.01)
                
                # 检查OpenCV窗口的键盘事件（允许用户中断）
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC或Q键
                    self.get_logger().info("用户中断等待")
                    return False
        else:
            self.get_logger().error("缺少坐标转换矩阵，无法等待车辆到达目标点")
            return False
    
    def plan_and_publish_path(self):
        """规划路径并发布到ROS话题"""
        # 读取location.txt文件中的位置
        locations = read_locations_from_file("location.txt")
        if not locations or len(locations) < 2:
            self.get_logger().error("无法从location.txt读取足够的位置信息")
            return
        
        # 确保QLabs转换矩阵已加载
        if self.qlabs_transformation_matrix is None:
            self.get_logger().error("QLabs坐标转换矩阵未加载，无法继续")
            return
        
        
            
        os.system(f"ros2 param set qcar2_hardware led_color_id 0") # 设置LED颜色
        
        # 三个阶段的规划和执行
        for stage in range(3):
            # 从ROS获取当前位置
            vehicle_pose = None
            if self.transformation_matrix is not None:
                try:
                    # 查询从map到base_scan的变换
                    trans = self.tf_buffer.lookup_transform(
                        'map',
                        'base_scan',
                        Time())
                    
                    # 提取位置和朝向
                    ros_x = trans.transform.translation.x
                    ros_y = trans.transform.translation.y
                    
                    # 从四元数中获取yaw角度
                    q = trans.transform.rotation
                    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                    
                    # 应用角度调整
                    adjusted_yaw = yaw + ANGLE_ADJUSTMENT
                    
                    # 将ROS坐标转换为像素坐标
                    pixel_position = ros_to_pixel_coordinates(ros_x, ros_y, self.transformation_matrix)
                    
                    vehicle_pose = {
                        "position": pixel_position,
                        "angle": adjusted_yaw
                    }
                    self.get_logger().info(f"获取到当前位置: ROS坐标({ros_x:.2f}, {ros_y:.2f}) => 像素坐标{pixel_position}")
                
                except TransformException as ex:
                    self.get_logger().warning(f'无法获取变换: {ex}')
            
            # 如果无法获取位置，则使用用户交互选择
            if vehicle_pose is None:
                self.get_logger().info("使用用户交互选择起点")
                vehicle_pose = get_user_pose("Choose Start Point", self.map_image, "车辆")
                if vehicle_pose is None:
                    self.get_logger().error("未选择车辆位姿")
                    return
            
            if stage < 2:
                # 第一和第二阶段：使用location.txt中的坐标
                qlabs_x, qlabs_y = locations[stage]
                self.get_logger().info(f"阶段 {stage+1}/3: 使用QLabs坐标 ({qlabs_x:.3f}, {qlabs_y:.3f})")
                
                # 将QLabs坐标转换为像素坐标
                goal_point = qlabs_to_pixel_coordinates(qlabs_x, qlabs_y, self.qlabs_transformation_matrix)
            else:
                # 第三阶段：使用map_transform.yaml中的(0,0)作为终点
                self.get_logger().info(f"阶段 {stage+1}/3: 使用原点(0.0, 0.0)作为目标点")
                goal_point = ros_to_pixel_coordinates(0.0, 0.0, self.transformation_matrix)
            
            self.get_logger().info(f"目标点像素坐标: {goal_point}")
            
            # 确保目标点在可通行区域内
            if not (0 <= goal_point[0] < self.traversable.shape[0] and 0 <= goal_point[1] < self.traversable.shape[1]):
                self.get_logger().error(f"目标点 {goal_point} 超出地图范围")
                continue
            
            if not self.traversable[goal_point]:
                self.get_logger().error(f"目标点 {goal_point} 不在可通行区域内")
                continue
            
            # 使用参考路径A*算法规划路径
            self.get_logger().info("使用参考路径进行路径规划...")
            path = a_star_with_reference_paths(
                vehicle_pose["position"], goal_point, 
                self.traversable, self.cost_map, self.yellow_line, 
                self.reference_paths
            )
            
            if path:
                self.current_path = path
                # 可视化路径
                # result_image = visualize_path(self.map_image, path, vehicle_pose)
                # stage_title = f"Path Planning Result - Stage {stage+1}/3"
                # cv2.imshow(stage_title, result_image)
                
                # 发布路径到ROS话题
                self.publish_path(path, vehicle_pose)
                self.get_logger().info(f"阶段 {stage+1}/3 路径规划完成并发布")
                
                # 在阶段间暂停3秒
                if stage < 2:  # 只在前两个阶段后暂停
                    self.get_logger().info(f"等待车辆到达目标点...")
                    self.wait_for_goal_reached(goal_point)  # 0.2米阈值
                    
                    
                    os.system(f"ros2 param set qcar2_hardware led_color_id {stage+1}")  # 设置LED颜色
                    self.get_logger().info(f"LED颜色改变，等待3秒后进入下一阶段...")
                    time.sleep(3.0)
            else:
                self.get_logger().error(f"阶段 {stage+1}/3 无法找到路径")
    
    def publish_path(self, path, vehicle_pose):
        """将像素路径转换为ROS路径消息并发布"""
        if not path:
            return
        
        # 创建Path消息
        ros_path = Path()
        ros_path.header = Header()
        ros_path.header.frame_id = "map"
        ros_path.header.stamp = self.get_clock().now().to_msg()
        
        # 如果有变换矩阵，将像素坐标转换为ROS坐标
        if self.transformation_matrix is not None:
            for y, x in path:
                # 从(y,x)转换为(x,y)格式
                pixel_point = np.array([[[x, y]]], dtype=np.float32)
                # 使用变换矩阵将像素坐标转换为ROS坐标
                ros_point = cv2.perspectiveTransform(pixel_point, self.transformation_matrix)
                ros_x, ros_y = ros_point[0][0]
                
                # 创建PoseStamped消息
                pose = PoseStamped()
                pose.header = ros_path.header
                pose.pose.position.x = float(ros_x)
                pose.pose.position.y = float(ros_y)
                pose.pose.position.z = 0.0
                
                # 添加到路径中
                ros_path.poses.append(pose) # type: ignore
        else:
            # 如果没有变换矩阵，直接使用像素坐标作为ROS坐标（仅用于可视化）
            for i, (y, x) in enumerate(path):
                pose = PoseStamped()
                pose.header = ros_path.header
                pose.pose.position.x = float(x) / 100.0  # 缩放以适应ROS可视化
                pose.pose.position.y = float(y) / 100.0
                pose.pose.position.z = 0.0
                
                ros_path.poses.append(pose) # type: ignore
        
        # 发布路径
        self.path_publisher.publish(ros_path)

def main(args=None):
    """ROS主函数"""
    rclpy.init(args=args)
    
    # 创建路径规划节点
    path_planner = PathPlannerNode()
    
    # 启动节点并保持运行
    try:
        rclpy.spin(path_planner)
    except KeyboardInterrupt:
        pass
    finally:
        # 关闭节点
        path_planner.destroy_node()
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        # 如果未关闭ROS，则关闭
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
