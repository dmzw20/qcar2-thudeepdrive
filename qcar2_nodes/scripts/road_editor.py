import numpy as np
import cv2
import pickle
import os

class Road:
    def __init__(self, id, points, name=""):
        self.id = id
        self.points = points  # 道路点列表 [(y1, x1), (y2, x2),...]
        self.name = name if name else f"Road {id}"


class RoadEditor:
    def __init__(self, map_image):
        self.map_image = map_image.copy()
        self.display_image = map_image.copy()
        self.roads = {}
        self.current_road_id = 0
        self.current_road_points = []
        self.editing_mode = "add_road"  # 'add_road', 'edit_road'
        self.selected_road = None
        self.road_colors = {}  # 存储每条道路的颜色
        self.is_creating_road = False
        self.temp_point = None
        
    def start_editor(self):
        """启动道路编辑器"""
        cv2.namedWindow("Road Editor")
        cv2.setMouseCallback("Road Editor", self.mouse_callback)
        
        while True:
            # 刷新显示
            self.update_display()
            
            # 显示操作模式
            cv2.putText(self.display_image, f"Mode: {self.editing_mode}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if self.is_creating_road:
                cv2.putText(self.display_image, "Creating Road (Press Enter to finish)", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Road Editor", self.display_image)
            
            key = cv2.waitKey(20)
            if key == 27:  # ESC键退出
                break
            elif key == ord('a'):  # 'a'键切换到添加道路模式
                self.editing_mode = "add_road"
                print("切换到添加道路模式")
            elif key == ord('e'):  # 'e'键切换到编辑道路模式
                self.editing_mode = "edit_road"
                print("切换到编辑道路模式")
            elif key == ord('s'):  # 's'键保存道路定义
                self.save_roads()
            elif key == ord('l'):  # 'l'键加载道路定义
                self.load_roads()
            elif key == 13:  # Enter键完成当前道路创建
                if self.is_creating_road and len(self.current_road_points) >= 2:
                    self.finish_current_road()
            elif key == 8:  # Backspace键删除最后一个点
                if self.current_road_points:
                    self.current_road_points.pop()
                    print("删除最后一个点")
            elif key == ord('d'):  # 'd'键删除当前选中的道路
                if self.editing_mode == "edit_road" and self.selected_road is not None:
                    if self.ask_yes_no(f"Are you sure you want to delete {self.roads[self.selected_road].name}?"):
                        deleted_name = self.roads[self.selected_road].name
                        del self.roads[self.selected_road]
                        print(f"已删除道路: {deleted_name}")
                        self.selected_road = None
                    
        cv2.destroyAllWindows()
        return self.roads
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if self.editing_mode == "add_road":
            if event == cv2.EVENT_LBUTTONDOWN:
                # 开始创建新道路或添加点到当前道路
                point = (y, x)
                if not self.is_creating_road:
                    self.is_creating_road = True
                    self.current_road_points = [point]
                    print(f"开始创建新道路，第一个点: {point}")
                else:
                    self.current_road_points.append(point)
                    print(f"添加点: {point}")
            
            elif event == cv2.EVENT_MOUSEMOVE and self.is_creating_road:
                # 实时显示可能的下一个点
                self.temp_point = (y, x)
                        
        elif self.editing_mode == "edit_road":
            if event == cv2.EVENT_LBUTTONDOWN:
                # 选择要编辑的道路
                clicked_road = self.find_road_at_point((y, x))
                if clicked_road is not None:
                    self.selected_road = clicked_road
                    print(f"选择编辑道路 {self.selected_road}")
    
    def update_display(self):
        """更新显示图像"""
        self.display_image = self.map_image.copy()
        
        # 绘制所有道路
        for road_id, road in self.roads.items():
            # 为每条道路分配唯一颜色（如果还没有）
            if road_id not in self.road_colors:
                self.road_colors[road_id] = tuple(np.random.randint(0, 255, 3).tolist())
            
            color = self.road_colors[road_id]
            
            # 绘制道路线
            for i in range(len(road.points) - 1):
                p1 = road.points[i]
                p2 = road.points[i + 1]
                cv2.line(self.display_image, (p1[1], p1[0]), (p2[1], p2[0]), color, 2)
            

            p1 = road.points[-1]
            p2 = road.points[0]
            cv2.line(self.display_image, (p1[1], p1[0]), (p2[1], p2[0]), color, 2)
            
            # 绘制道路点
            for p in road.points:
                cv2.circle(self.display_image, (p[1], p[0]), 3, color, -1)
            
            # 显示道路ID和名称
            if road.points:
                pos = road.points[0]
                cv2.putText(self.display_image, f"{road.name} (ID:{road.id})", 
                            (pos[1], pos[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 绘制正在创建的道路
        if self.is_creating_road and self.current_road_points:
            # 给当前道路一个临时颜色
            temp_color = (0, 255, 255)  # 黄色
            
            # 绘制已添加的点和线
            for i in range(len(self.current_road_points)):
                p = self.current_road_points[i]
                cv2.circle(self.display_image, (p[1], p[0]), 3, temp_color, -1)
                if i > 0:
                    p_prev = self.current_road_points[i-1]
                    cv2.line(self.display_image, (p_prev[1], p_prev[0]), (p[1], p[0]), temp_color, 2)
            
            # 绘制临时线（从最后一个点到鼠标位置）
            if self.temp_point and len(self.current_road_points) > 0:
                p_last = self.current_road_points[-1]
                cv2.line(self.display_image, (p_last[1], p_last[0]), 
                         (self.temp_point[1], self.temp_point[0]), (128, 128, 128), 1)
        
        # 显示选中的道路（高亮显示）
        if self.selected_road is not None:
            road = self.roads.get(self.selected_road)
            if road:
                for i in range(len(road.points) - 1):
                    p1 = road.points[i]
                    p2 = road.points[i + 1]
                    cv2.line(self.display_image, (p1[1], p1[0]), (p2[1], p2[0]), (0, 0, 255), 3)
                
                # 连接首尾
                p1 = road.points[-1]
                p2 = road.points[0]
                cv2.line(self.display_image, (p1[1], p1[0]), (p2[1], p2[0]), (0, 0, 255), 3)
    
    def finish_current_road(self):
        """完成当前道路创建"""
        if len(self.current_road_points) >= 2:
            
            # 询问道路名称
            road_name = self.ask_road_name()
            
            # 创建新道路
            new_road = Road(
                id=self.current_road_id,
                points=self.current_road_points.copy(),
                name=road_name
            )
            
            # 添加到道路字典
            self.roads[self.current_road_id] = new_road
            print(f"道路 {self.current_road_id} 创建完成，包含 {len(self.current_road_points)} 个点")
            
            # 增加道路ID计数
            self.current_road_id += 1
            
            # 重置当前道路点
            self.current_road_points = []
            self.is_creating_road = False
    
    def find_road_at_point(self, point, threshold=10):
        """查找给定点附近的道路"""
        min_dist = float('inf')
        closest_road = None
        
        for road_id, road in self.roads.items():
            for i in range(len(road.points) - 1):
                p1 = np.array(road.points[i])
                p2 = np.array(road.points[i + 1])
                
                # 计算点到线段的距离
                line_vec = p2 - p1
                point_vec = np.array(point) - p1
                line_len = np.linalg.norm(line_vec)
                
                if line_len == 0:
                    continue
                
                # 计算投影比例
                proj_ratio = np.dot(point_vec, line_vec) / (line_len * line_len)
                
                if 0 <= proj_ratio <= 1:
                    # 点在线段投影上
                    proj_point = p1 + proj_ratio * line_vec
                    dist = np.linalg.norm(np.array(point) - proj_point)
                    
                    if dist < min_dist and dist < threshold:
                        min_dist = dist
                        closest_road = road_id
                
                # 也检查道路端点
                dist_to_p1 = np.linalg.norm(np.array(point) - p1)
                dist_to_p2 = np.linalg.norm(np.array(point) - p2)
                
                if dist_to_p1 < min_dist and dist_to_p1 < threshold:
                    min_dist = dist_to_p1
                    closest_road = road_id
                
                if dist_to_p2 < min_dist and dist_to_p2 < threshold:
                    min_dist = dist_to_p2
                    closest_road = road_id
        
        return closest_road
    
    def ask_yes_no(self, question):
        """显示是/否对话框"""
        dialog_image = np.zeros((150, 300, 3), dtype=np.uint8) + 200  # 灰色背景
        cv2.putText(dialog_image, question, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(dialog_image, "Y/y - Yes, N/n - No", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        cv2.imshow("Question", dialog_image)
        while True:
            key = cv2.waitKey(0)
            if key == ord('y') or key == ord('Y'):
                cv2.destroyWindow("Question")
                return True
            elif key == ord('n') or key == ord('N'):
                cv2.destroyWindow("Question")
                return False
    
    def ask_road_name(self):
        """询问道路名称"""
        dialog_image = np.zeros((150, 400, 3), dtype=np.uint8) + 200  # 灰色背景
        cv2.putText(dialog_image, "Input road name (in terminal)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.imshow("Road Name", dialog_image)
        cv2.waitKey(1)
        
        print("\n请输入道路名称:")
        name = input()
        
        cv2.destroyWindow("Road Name")
        return name if name else f"Road {self.current_road_id}"
    
    def save_roads(self):
        """保存道路定义"""
        try:
            with open('roads_definition.pkl', 'wb') as f:
                pickle.dump(self.roads, f)
            print(f"已保存 {len(self.roads)} 条道路定义到 roads_definition.pkl")
        except Exception as e:
            print(f"保存道路定义失败: {e}")
    
    def load_roads(self):
        """加载道路定义"""
        try:
            if os.path.exists('roads_definition.pkl'):
                with open('roads_definition.pkl', 'rb') as f:
                    self.roads = pickle.load(f)
                
                # 更新当前道路ID
                if self.roads:
                    self.current_road_id = max(self.roads.keys()) + 1
                
                print(f"已加载 {len(self.roads)} 条道路定义")
            else:
                print("未找到道路定义文件")
        except Exception as e:
            print(f"加载道路定义失败: {e}")


# 测试函数
def test_road_editor():
    map_path = "ori_map.png"  # 替换为您的地图路径
    map_image = cv2.imread(map_path)
    
    if map_image is None:
        print("无法读取地图图像")
        return
    
    editor = RoadEditor(map_image)
    roads = editor.start_editor()
    print(f"总共定义了 {len(roads)} 条道路")

if __name__ == "__main__":
    test_road_editor()