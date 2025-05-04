#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class ImageDebugNode(Node):
    def __init__(self):
        super().__init__('image_debug_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/color_image',
            self.image_callback,
            10)
        self.get_logger().info("开始监听 /camera/color_image 话题")

    def image_callback(self, msg):
        self.get_logger().info(f"收到图像消息: 编码={msg.encoding}, 尺寸={msg.width}x{msg.height}")
        
        # 输出消息的详细信息
        self.get_logger().info(f"消息头: frame_id={msg.header.frame_id}")
        self.get_logger().info(f"步长: {msg.step}, 数据长度: {len(msg.data)}")
        
        # 数据非空检查
        if len(msg.data) == 0:
            self.get_logger().error("图像数据为空!")
            return
        
        # 检查数据量是否符合预期
        expected_data_size = msg.height * msg.step
        if len(msg.data) != expected_data_size:
            self.get_logger().warn(f"图像数据大小不匹配! 期望: {expected_data_size}, 实际: {len(msg.data)}")
        
        try:
            # 尝试转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # 检查转换后的图像是否为空
            if cv_image is None or cv_image.size == 0:
                self.get_logger().error("转换后的OpenCV图像为空!")
                return
                
            self.get_logger().info(f"OpenCV图像形状: {cv_image.shape}, 类型: {cv_image.dtype}")
            
            # 显示图像
            cv2.imshow("Image Debug", cv_image)
            cv2.waitKey(1)
            
            # 保存图像用于进一步分析
            cv2.imwrite("/workspaces/isaac_ros-dev/ros2/src/qcar2_nodes/scripts/debug_image/debug_image.jpg", cv_image)
            self.get_logger().info("已保存调试图像到 /workspaces/isaac_ros-dev/ros2/src/qcar2_nodes/scripts/debug_image/debug_image.jpg")
            
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge错误: {e}")
        except Exception as e:
            self.get_logger().error(f"处理图像时发生错误: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageDebugNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
