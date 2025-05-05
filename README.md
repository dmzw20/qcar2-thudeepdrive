# qcar2-thudeepdrive

This repository was created to submit entries for the ACC Competition 2025.

## Dependency

At the bottom of the `Dockerfile.quanser` add your Python packages as shown below:

```
RUN pip3 install typing-extensions==4.4.0 -U \
    ultralytics==8.3.91 \
    lap==0.4.0
```

## Usage

First, you need to run in the virtual-qcar2 docker container (Our virtual-qcar2 image is a very old version, and I'm not sure if this will cause any issues):

```
cd /home/qcar2_scripts/python/Base_Scenarios_Python/ && python3 Setup_Real_Scenario.py
```

Then open four terminals in Isaac-ROS, run `source install/setup.bash` in each, and enter the following commands:

```
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py
cd /workspaces/isaac_ros-dev/ros2/src/qcar2_nodes/scripts/ && python path_follow.py
cd /workspaces/isaac_ros-dev/ros2/src/qcar2_nodes/scripts/ && python yolo_detect.py
cd /workspaces/isaac_ros-dev/ros2/src/qcar2_nodes/scripts/ && python pre_defined_astar_planner.py
```

Click on the OpenCV window that appears and press the 'p' key to start.

## Youtube Link

https://youtu.be/nUBf36-EFMY
