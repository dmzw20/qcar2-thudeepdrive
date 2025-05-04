import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion
import math
import os
import pickle

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
    
    # cv2.imshow("traversable", traversable.astype(np.uint8) * 255)
    # cv2.imshow("yellow line", yellow.astype(np.uint8) * 255)
    # cv2.imshow("zebra crossings", zebra_crossings.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    
    
    # 创建成本地图
    cost_map = np.zeros_like(traversable, dtype=float)
    
    # 斑马线区域增加成本（需要停车等待）    
    cost_map[blue_mask > 0] += 2
    
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

def create_reference_paths(map_image, load_existing=False):
    """允许用户手动绘制参考路径和方向，可选择加载现有路径"""
    reference_paths = []
    
    # 加载现有路径（如果需要）
    if load_existing:
        try:
            if os.path.exists('reference_paths.pkl'):
                with open('reference_paths.pkl', 'rb') as f:
                    reference_paths = pickle.load(f)
                print(f"已加载 {len(reference_paths)} 条现有参考路径")
            else:
                print("未找到现有参考路径文件，将创建新文件")
        except Exception as e:
            print(f"加载参考路径失败: {e}")
    
    drawing = False
    current_path = []
    selected_path_index = -1  # 追踪当前选中的路径索引，-1表示未选中
    
    def draw_path(event, x, y, flags, param):
        nonlocal drawing, current_path, selected_path_index
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 如果正在绘制，则添加点到当前路径
            if drawing:
                # 添加新点到当前路径
                current_path.append((y, x))
                # 绘制线段
                if len(current_path) > 1:
                    p1_y, p1_x = current_path[-2]
                    p2_y, p2_x = current_path[-1]
                    
                    # 计算线段长度
                    line_length = np.sqrt((p2_y - p1_y)**2 + (p2_x - p1_x)**2)
                    
                    # 优化的箭头显示逻辑
                    if line_length > 50:  # 如果线段较长
                        # 绘制基础线
                        cv2.line(param["display_image"], (p1_x, p1_y), (p2_x, p2_y), (0, 255, 0), 2)
                        
                        # 沿线段绘制多个小箭头
                        num_arrows = max(2, int(line_length / 50))  # 每50像素一个箭头，至少2个
                        for i in range(1, num_arrows):
                            ratio = i / num_arrows
                            mid_y = int(p1_y + (p2_y - p1_y) * ratio)
                            mid_x = int(p1_x + (p2_x - p1_x) * ratio)
                            
                            # 计算箭头方向
                            arrow_length = min(20, line_length / num_arrows * 0.8)
                            dir_y = (p2_y - p1_y) / line_length
                            dir_x = (p2_x - p1_x) / line_length
                            
                            # 绘制小箭头
                            end_y = int(mid_y + dir_y * arrow_length / 2)
                            end_x = int(mid_x + dir_x * arrow_length / 2)
                            start_y = int(mid_y - dir_y * arrow_length / 2)
                            start_x = int(mid_x - dir_x * arrow_length / 2)
                            
                            cv2.arrowedLine(param["display_image"], 
                                          (start_x, start_y), (end_x, end_y),
                                          (0, 255, 0), 2, tipLength=0.3)
                    else:
                        # 短线段使用单一箭头
                        cv2.arrowedLine(param["display_image"], (p1_x, p1_y), (p2_x, p2_y), 
                                      (0, 255, 0), 2, tipLength=min(0.3, 15/line_length if line_length > 0 else 0.3))
                    
                    cv2.imshow("Create Reference Path", param["display_image"])
                    print(f"添加点: ({y}, {x})")
            else:
                # 如果没有正在绘制的路径，检查是否点击了某条路径
                clicked_on_path = False
                for idx, path in enumerate(reference_paths):
                    for i in range(len(path)-1):
                        p1_y, p1_x = path[i]
                        p2_y, p2_x = path[i+1]
                        
                        # 计算点到线段的距离
                        line_length = np.sqrt((p2_y - p1_y)**2 + (p2_x - p1_x)**2)
                        if line_length > 0:
                            # 计算点到线段的距离
                            t = ((y - p1_y)*(p2_y - p1_y) + (x - p1_x)*(p2_x - p1_x)) / (line_length**2)
                            t = max(0, min(1, t))  # 限制t在[0,1]范围内
                            proj_y = p1_y + t * (p2_y - p1_y)
                            proj_x = p1_x + t * (p2_x - p1_x)
                            distance = np.sqrt((y - proj_y)**2 + (x - proj_x)**2)
                            
                            if distance < 5:  # 如果点击点靠近某条线段
                                selected_path_index = idx
                                clicked_on_path = True
                                print(f"已选中路径 {selected_path_index+1}")
                                
                                # 重绘所有路径，突出显示选中的路径
                                redraw_all_paths(param["display_image"], reference_paths, selected_path_index)
                                break
                    
                    if clicked_on_path:
                        break
                
                # 如果没有点击任何路径，则开始新的路径绘制
                if not clicked_on_path:
                    drawing = True
                    current_path = [(y, x)]  # 使用(y,x)格式
                    selected_path_index = -1  # 取消当前选中
                    param["display_image"] = map_image.copy()
                    redraw_all_paths(param["display_image"], reference_paths, selected_path_index)
                    print(f"开始新路径，第一点: ({y}, {x})")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 结束当前路径，将其添加到参考路径列表
            if drawing and len(current_path) > 1:
                reference_paths.append(current_path)
                print(f"路径 {len(reference_paths)} 已添加，包含 {len(current_path)} 个点")
                drawing = False
                current_path = []
                selected_path_index = -1  # 取消当前选中
                
                # 更新显示图像，显示所有路径
                param["display_image"] = map_image.copy()
                redraw_all_paths(param["display_image"], reference_paths, selected_path_index)
    
    def redraw_all_paths(display_image, paths, selected_idx=-1):
        """重绘所有路径，突出显示选中路径"""
        for idx, path in enumerate(paths):
            # 为选中路径使用不同颜色
            color = (0, 0, 255) if idx == selected_idx else (0, 200, 0)  # 选中路径用红色，其他为绿色
            
            for i in range(len(path)-1):
                p1_y, p1_x = path[i]
                p2_y, p2_x = path[i+1]
                
                # 计算线段长度
                line_length = np.sqrt((p2_y - p1_y)**2 + (p2_x - p1_x)**2)
                
                # 优化的箭头显示逻辑（与上面相同）
                if line_length > 50:  # 如果线段较长
                    # 绘制基础线
                    cv2.line(display_image, (p1_x, p1_y), (p2_x, p2_y), color, 2)
                    
                    # 沿线段绘制多个小箭头
                    num_arrows = max(2, int(line_length / 50))  # 每50像素一个箭头，至少2个
                    for j in range(1, num_arrows):
                        ratio = j / num_arrows
                        mid_y = int(p1_y + (p2_y - p1_y) * ratio)
                        mid_x = int(p1_x + (p2_x - p1_x) * ratio)
                        
                        # 计算箭头方向
                        arrow_length = min(20, line_length / num_arrows * 0.8)
                        dir_y = (p2_y - p1_y) / line_length
                        dir_x = (p2_x - p1_x) / line_length
                        
                        # 绘制小箭头
                        end_y = int(mid_y + dir_y * arrow_length / 2)
                        end_x = int(mid_x + dir_x * arrow_length / 2)
                        start_y = int(mid_y - dir_y * arrow_length / 2)
                        start_x = int(mid_x - dir_x * arrow_length / 2)
                        
                        cv2.arrowedLine(display_image, 
                                      (start_x, start_y), (end_x, end_y),
                                      color, 2, tipLength=0.3)
                else:
                    # 短线段使用单一箭头
                    cv2.arrowedLine(display_image, (p1_x, p1_y), (p2_x, p2_y), 
                                  color, 2, tipLength=min(0.3, 15/line_length if line_length > 0 else 0.3))
        
        cv2.imshow("Create Reference Path", display_image)
    
    # 创建窗口并设置鼠标回调
    cv2.namedWindow("Create Reference Path")
    params = {"display_image": map_image.copy()}
    cv2.setMouseCallback("Create Reference Path", draw_path, params)
    
    # 如果加载了现有路径，先绘制它们
    if load_existing and reference_paths:
        redraw_all_paths(params["display_image"], reference_paths)
    
    # 显示初始图像和说明
    cv2.imshow("Create Reference Path", params["display_image"])
    print("操作说明:")
    print("左键点击空白处: 开始新路径/添加路径点")
    print("左键点击已有路径: 选中该路径")
    print("右键点击: 完成当前路径")
    print("按 'r' 清除当前路径")
    print("按 'd' 删除选中的路径")
    print("按 'c' 清除所有路径")
    print("按 's' 保存所有路径")
    print("按 'ESC' 或 'q' 退出")
    
    # 等待用户操作
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC或q键退出
            break
        elif key == ord('c'):  # c键清除所有路径
            reference_paths = []
            drawing = False
            current_path = []
            selected_path_index = -1
            params["display_image"] = map_image.copy()
            cv2.imshow("Create Reference Path", params["display_image"])
            print("所有路径已清除")
        elif key == ord('r'):  # r键清除当前路径
            if drawing and len(current_path) > 0:
                drawing = False
                current_path = []
                # 只重绘已保存的路径
                params["display_image"] = map_image.copy()
                redraw_all_paths(params["display_image"], reference_paths, selected_path_index)
                print("当前路径已清除")
        elif key == ord('d'):  # d键删除选中的路径
            if selected_path_index >= 0 and selected_path_index < len(reference_paths):
                print(f"删除路径 {selected_path_index+1}")
                reference_paths.pop(selected_path_index)
                selected_path_index = -1  # 取消选中
                params["display_image"] = map_image.copy()
                redraw_all_paths(params["display_image"], reference_paths)
        elif key == ord('s'):  # s键保存路径
            if len(reference_paths) > 0:
                with open('reference_paths.pkl', 'wb') as f:
                    pickle.dump(reference_paths, f)
                print(f"已保存 {len(reference_paths)} 条参考路径到 reference_paths.pkl")
    
    cv2.destroyAllWindows()
    return reference_paths

def main():
    # 读取地图图像
    map_path = "ori_map.png"
    map_image = cv2.imread(map_path)
    
    if map_image is None:
        print("无法读取地图图像")
        return
    
    # 选择模式
    print("请选择操作模式:")
    print("1. 创建新的参考路径")
    print("2. 加载并编辑现有参考路径")
    mode = input("请输入操作模式 (1/2): ")
    
    if mode == '1':
        # 创建新的参考路径模式
        reference_paths = create_reference_paths(map_image, load_existing=False)
    elif mode == '2':
        # 加载并编辑现有参考路径
        reference_paths = create_reference_paths(map_image, load_existing=True)

if __name__ == "__main__":
    main()
