#!/usr/bin/env python3
import numpy as np
import cv2
import os
import pickle
import math
import yaml
from matplotlib import pyplot as plt

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

def collect_calibration_points(map_image, num_points=4):
    """收集地图上的标定点和对应的实际位置坐标"""
    print("=== 地图坐标系与实际坐标系转换矩阵标定 ===")
    print(f"请依次选择{num_points}个标定点，并输入对应的实际ROS坐标")
    
    # 用于存储像素坐标和实际坐标的对应点
    pixel_points = []
    world_points = []
    
    # 复制地图用于显示标记点
    display_image = map_image.copy()
    
    for i in range(num_points):
        # 获取地图上的点击位置
        point_message = f"标定点 {i+1}/{num_points}"
        pixel_point = get_user_click_position("Select Calibration Point", display_image, point_message)
        
        if pixel_point is None:
            print("未选择点，退出标定")
            return None, None
        
        # 标记已选择的点
        y, x = pixel_point # type: ignore
        cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(display_image, str(i+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 添加到像素坐标列表
        pixel_points.append(pixel_point)
        
        # 输入对应的实际坐标
        while True:
            try:
                x_world = float(input(f"请输入标定点 {i+1} 在ROS坐标系中的X坐标: "))
                y_world = float(input(f"请输入标定点 {i+1} 在ROS坐标系中的Y坐标: "))
                world_points.append((x_world, y_world))
                break
            except ValueError:
                print("输入无效，请重新输入数字")
    
    return np.array(pixel_points), np.array(world_points)

def compute_transformation_matrix(pixel_points, world_points):
    """计算从地图像素坐标到实际坐标的转换矩阵"""
    # 检查点的数量是否足够
    if len(pixel_points) < 3 or len(world_points) < 3:
        print("点数不足，至少需要3个点来计算转换矩阵")
        return None
    
    # 从像素坐标(y,x)转换为OpenCV使用的(x,y)格式
    cv_pixel_points = np.array([(x, y) for y, x in pixel_points], dtype=np.float32)
    
    # 世界坐标
    cv_world_points = np.array(world_points, dtype=np.float32)
    
    # 计算仿射变换矩阵
    matrix, _ = cv2.findHomography(cv_pixel_points, cv_world_points)
    
    # 返回转换矩阵（用于从像素坐标转换为实际坐标）
    return matrix

def test_transformation(map_image, transformation_matrix):
    """测试转换矩阵的准确性"""
    print("\n=== 转换矩阵测试 ===")
    print("请在地图上点击任意位置，程序将显示对应的ROS坐标")
    
    while True:
        # 获取地图上的点击位置
        point = get_user_click_position("Test Transformation", map_image, "测试点")
        
        if point is None:
            break
        
        # 将点转换为CV坐标格式
        y, x = point # type: ignore
        cv_point = np.array([[x, y]], dtype=np.float32)
        
        # 应用变换矩阵
        transformed_point = cv2.perspectiveTransform(cv_point.reshape(-1, 1, 2), transformation_matrix)
        ros_x, ros_y = transformed_point[0][0]
        
        print(f"地图像素坐标: ({y}, {x}) => ROS坐标: ({ros_x:.3f}, {ros_y:.3f})")
        
        # 显示带有标记的地图
        test_image = map_image.copy()
        cv2.circle(test_image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(test_image, f"({ros_x:.2f}, {ros_y:.2f})", (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Test Transformation", test_image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 按ESC键退出测试
        if key == 27:  # ESC键
            break
            
    print("测试结束")

def visualize_calibration(map_image, pixel_points, world_points, transformation_matrix):
    """可视化标定点和转换结果"""
    # 创建可视化图像
    visual_img = map_image.copy()
    
    # 绘制标定点
    for i, (p_pixel, p_world) in enumerate(zip(pixel_points, world_points)):
        y, x = p_pixel
        wx, wy = p_world
        
        # 绘制标定点
        cv2.circle(visual_img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(visual_img, f"{i+1}: ({wx:.2f}, {wy:.2f})", (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 在图像上绘制网格，用于验证转换的准确性
    grid_step = 50  # 像素网格步长
    rows, cols = map_image.shape[:2]
    
    for y in range(0, rows, grid_step):
        for x in range(0, cols, grid_step):
            # 转换网格点到ROS坐标
            cv_point = np.array([[x, y]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(cv_point.reshape(-1, 1, 2), transformation_matrix)
            ros_x, ros_y = transformed_point[0][0]
            
            # 绘制小网格点和坐标值
            cv2.circle(visual_img, (x, y), 2, (255, 0, 0), -1)
            if x % (grid_step * 2) == 0 and y % (grid_step * 2) == 0:  # 减少标签数量
                cv2.putText(visual_img, f"({ros_x:.1f}, {ros_y:.1f})", (x+3, y+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    # 显示可视化结果
    cv2.imshow("Calibration Visualization", visual_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存可视化结果
    cv2.imwrite("calibration_visualization.png", visual_img)
    print("已保存标定可视化结果到 calibration_visualization.png")

def save_transformation_matrix(transformation_matrix, output_file="map_transform.yaml"):
    """保存转换矩阵到YAML文件"""
    # 将转换矩阵转换为可序列化的格式
    matrix_list = transformation_matrix.tolist()
    
    # 创建数据字典
    data = {
        'transformation_matrix': matrix_list,
        'description': 'Transformation matrix from map pixel coordinates to ROS world coordinates'
    }
    
    # 保存到YAML文件
    with open(output_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=None)
    
    print(f"转换矩阵已保存到 {output_file}")

def main():
    """主函数：执行地图到实际坐标的标定过程"""
    # 读取地图图像
    map_path = "ori_map.png"
    map_image = cv2.imread(map_path)
    
    if map_image is None:
        print(f"无法读取地图图像: {map_path}")
        return
    
    # 收集标定点
    pixel_points, world_points = collect_calibration_points(map_image, num_points=6)
    
    if pixel_points is None or world_points is None:
        print("标定点收集失败")
        return
    
    # 计算转换矩阵
    transformation_matrix = compute_transformation_matrix(pixel_points, world_points)
    
    if transformation_matrix is None:
        print("无法计算转换矩阵")
        return
    
    print("\n计算得到的转换矩阵:")
    print(transformation_matrix)
    
    # 可视化标定结果
    visualize_calibration(map_image, pixel_points, world_points, transformation_matrix)
    
    # 测试转换矩阵
    test_transformation(map_image, transformation_matrix)
    
    # 保存转换矩阵
    save_transformation_matrix(transformation_matrix)
    
    print("\n标定完成！")

if __name__ == "__main__":
    main()
