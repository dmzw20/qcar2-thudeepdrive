#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
import sys
import time
from std_msgs.msg import String
from qcar2_interfaces.msg import MotorCommands
from ultralytics import YOLO

class YoloDetector(Node):
    """ROS2节点：订阅相机图像，使用YOLOv8m进行目标检测，并发布处理后的图像"""

    def __init__(self):
        super().__init__('yolo_detector')

        # 声明参数
        self.declare_parameter('confidence_threshold', 0.73)  # 置信度阈值
        self.declare_parameter('weights_path', '/workspaces/isaac_ros-dev/ros2/src/qcar2_nodes/scripts/best.pt')  # 模型权重路径
        self.declare_parameter('traffic_light_topic', '/traffic_light_state')  # 红绿灯状态话题
        self.declare_parameter('stop_sign_timeout', 0.5)  # 停车标志超时时间(秒)
        self.declare_parameter('stop_command_duration', 1.0)  # 停车命令持续时间(秒)
        
        # 获取参数
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.weights_path = self.get_parameter('weights_path').value
        self.traffic_light_topic = self.get_parameter('traffic_light_topic').value
        self.stop_sign_timeout = self.get_parameter('stop_sign_timeout').value
        self.stop_command_duration = self.get_parameter('stop_command_duration').value
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 创建订阅者和发布者
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/color_image',  # 订阅的图像话题
            self.image_callback,
            10)
        
        self.detection_publisher = self.create_publisher(
            Image,
            '/yolo_image',  # 发布处理后图像的话题
            10)
        
        # 创建停车命令发布器
        self.motor_publisher = self.create_publisher(
            MotorCommands,
            '/qcar2_motor_speed_cmd',  # 电机控制话题
            10)
        
        # 创建红绿灯状态发布器
        self.traffic_light_publisher = self.create_publisher(
            String,
            self.traffic_light_topic,  # 红绿灯状态话题
            10)
        
        # 初始化停车标志相关变量
        self.last_stop_sign_time = 0.0  # 上次检测到停车标志的时间
        self.stop_sign_detected = False  # 是否检测到停车标志
        self.stop_command_active = False  # 是否正在发送停车命令
        self.stop_command_start_time = 0.0  # 开始发送停车命令的时间
        
        # 创建停车标志检查的定时器 (100Hz，确保快速响应)
        self.stop_check_timer = self.create_timer(0.01, self.check_stop_sign_status)
        
        # 加载YOLO模型
        self.model = self.load_model()
        
        # 初始化发布红绿灯状态为'none'
        self.publish_traffic_light_state('none')
        
        self.get_logger().info('YOLOv8m检测节点已初始化')
        self.get_logger().info(f'模型权重: {self.weights_path}')
        self.get_logger().info(f'置信度阈值: {self.conf_threshold}')
        self.get_logger().info(f'红绿灯状态话题: {self.traffic_light_topic}')
        self.get_logger().info(f'停车标志超时时间: {self.stop_sign_timeout}秒')
        self.get_logger().info(f'停车命令持续时间: {self.stop_command_duration}秒')

    def load_model(self):
        """加载YOLOv8模型"""
        try:
            # 确保权重文件存在
            if not os.path.exists(self.weights_path):
                self.get_logger().error(f'找不到权重文件: {self.weights_path}')
                raise FileNotFoundError(f'找不到权重文件: {self.weights_path}')
            
            # 加载YOLOv8模型
            self.get_logger().info('正在加载YOLOv8模型...')
            model = YOLO(self.weights_path)
            
            self.get_logger().info('YOLOv8模型加载成功')
            return model
            
        except Exception as e:
            self.get_logger().error(f'加载模型时出错: {str(e)}')
            sys.exit(1)

    def publish_traffic_light_state(self, state):
        """发布红绿灯状态"""
        msg = String()
        msg.data = state
        self.traffic_light_publisher.publish(msg)
        if state != 'none':
            self.get_logger().info(f'检测到红绿灯，状态: {state}')

    def send_stop_command(self):
        """发送停车命令"""
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [0.0, 0.0]  # 转向角和油门均为0
        self.motor_publisher.publish(msg)
        self.get_logger().debug('发送停车命令')

    def check_stop_sign_status(self):
        """检查停车标志状态并控制停车命令发送"""
        current_time = time.time()
        
        # 检查是否需要开始发送停车命令
        if self.stop_sign_detected and not self.stop_command_active:
            # 检查是否超过超时时间
            if current_time - self.last_stop_sign_time > self.stop_sign_timeout:
                # 开始发送停车命令
                self.stop_command_active = True
                self.stop_command_start_time = current_time
                self.get_logger().info(f'停车标志超时 ({self.stop_sign_timeout}秒)，开始发送停车命令')
                self.send_stop_command()
        
        # 如果正在发送停车命令，则检查是否需要继续发送
        elif self.stop_command_active:
            # 检查是否已经发送足够长时间
            if current_time - self.stop_command_start_time < self.stop_command_duration:
                # 继续发送停车命令
                self.send_stop_command()
            else:
                # 停止发送停车命令
                self.stop_command_active = False
                self.stop_sign_detected = False
                self.get_logger().info(f'已经发送停车命令 {self.stop_command_duration}秒，停止发送')

    def determine_traffic_light_color(self, box, image):
        """判断红绿灯颜色"""
        # 获取红绿灯边界框
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 提取红绿灯区域
        light_roi = image[y1:y2, x1:x2]
        
        if light_roi.size == 0:
            return 'unknown'
        
        # 转换到HSV颜色空间
        # hsv_roi = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
        
        # 定义颜色阈值
        # lower_red1 = np.array([0, 100, 100])
        # upper_red1 = np.array([10, 255, 255])
        # lower_red2 = np.array([160, 100, 100])
        # upper_red2 = np.array([180, 255, 255])
        lower_red = np.array([40, 40, 240])
        upper_red = np.array([160, 160, 255])
        lower_green = np.array([80, 240, 100])
        upper_green = np.array([180, 255, 180])
        lower_yellow = np.array([100, 230, 230])
        upper_yellow = np.array([150, 255, 255])
        
        # 创建颜色掩码
        # mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        # mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        # mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_red = cv2.inRange(light_roi, lower_red, upper_red)
        mask_green = cv2.inRange(light_roi, lower_green, upper_green)
        mask_yellow = cv2.inRange(light_roi, lower_yellow, upper_yellow)
        
        # 计算各颜色区域的像素数量
        red_count = cv2.countNonZero(mask_red)
        green_count = cv2.countNonZero(mask_green)
        yellow_count = cv2.countNonZero(mask_yellow)
        
        # 判断主要颜色
        max_count = max(red_count, green_count, yellow_count)
        
        if max_count < 10:  # 如果所有颜色的像素都很少，可能无法判断
            return 'unknown'
        elif red_count == max_count:
            return 'red'
        elif green_count == max_count:
            return 'green'
        elif yellow_count == max_count:
            return 'yellow'
        else:
            return 'unknown'

    def image_callback(self, msg):
        """处理接收的图像并进行目标检测"""
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 执行YOLOv8检测
            results = self.model(cv_image, conf=self.conf_threshold)
            
            # 初始化默认的红绿灯状态
            traffic_light_state = 'none'
            
            # 在图像上绘制检测结果
            annotated_img = results[0].plot()
            
            # 处理检测结果
            boxes = results[0].boxes
            if len(boxes) > 0:
                # 检查是否有停车标志和红绿灯
                stop_detected = False
                light_detected = False
                
                for box in boxes:
                    cls_id = int(box.cls.item())
                    class_name = results[0].names[cls_id]
                    confidence = float(box.conf.item())
                    
                    # 如果检测到停车标志，且标识的中心点在图像的右半部分
                    if class_name.lower() == "stop" and box.xyxy[0][0] > cv_image.shape[1] / 2:
                        self.get_logger().info(f'检测到停车标志，置信度: {confidence:.2f}')
                        # 更新最后检测时间和状态
                        self.last_stop_sign_time = time.time()
                        self.stop_sign_detected = True
                        stop_detected = True
                    
                    # 如果检测到红绿灯
                    elif class_name.lower() == "light":
                        light_detected = True
                        # 判断红绿灯颜色
                        color = self.determine_traffic_light_color(box, cv_image)
                        if color != 'unknown':
                            traffic_light_state = color
                            # 在图像上标记红绿灯颜色
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            color_bgr = (0, 0, 255) if color == 'red' else (0, 255, 0) if color == 'green' else (0, 255, 255)
                            cv2.putText(annotated_img, color, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
            
            # 发布红绿灯状态
            self.publish_traffic_light_state(traffic_light_state)
            
            # 将处理后的图像转换回ROS消息并发布
            detection_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
            detection_msg.header = msg.header  # 保持相同的时间戳和帧ID
            self.detection_publisher.publish(detection_msg)
            
            num_detections = len(results[0].boxes)
            if num_detections > 0:
                self.get_logger().info(f'检测到 {num_detections} 个目标')
            
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge错误: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'处理图像时出错: {str(e)}')

def main(args=None):
    # 初始化ROS
    rclpy.init(args=args)
    
    # 创建节点
    yolo_detector = YoloDetector()
    
    try:
        # 运行节点
        rclpy.spin(yolo_detector)
    except KeyboardInterrupt:
        # 处理Ctrl+C
        yolo_detector.get_logger().info('用户中断，正在关闭节点...')
    except Exception as e:
        # 处理其他异常
        yolo_detector.get_logger().error(f'发生异常: {str(e)}')
    finally:
        # 清理资源
        yolo_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
