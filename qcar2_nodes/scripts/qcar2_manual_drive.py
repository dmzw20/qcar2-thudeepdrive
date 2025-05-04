#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from qcar2_interfaces.msg import MotorCommands
import threading
import sys
import os
import termios
import tty
import time

class QCar2ManualDrive(Node):
    def __init__(self):
        super().__init__('qcar2_manual_drive')
        
        # Create publisher for motor commands
        self.publisher = self.create_publisher(
            MotorCommands, 
            '/qcar2_motor_speed_cmd', 
            # '/cmd_vel_nav',
            10)
        
        # Initialize control values
        self.steering_angle = 0.0  # radians
        self.motor_throttle = 0.0  # m/s
        
        # Control parameters
        self.max_steering = 0.6  # max steering angle in radians
        self.steering_increment = 0.05  # steering increment
        self.max_throttle = 1.0  # max throttle in m/s
        self.throttle_increment = 0.05  # throttle increment
        
        # Timer for publishing commands
        self.timer = self.create_timer(0.1, self.send_command)  # 10 Hz
        
        # Print control instructions
        self.print_instructions()
        
    def send_command(self):
        """Send the current steering and throttle commands"""
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [self.steering_angle, self.motor_throttle]
        self.publisher.publish(msg)
        # Clear line and move to start
        print(f"\rSteering: {self.steering_angle:.2f} rad, Throttle: {self.motor_throttle:.2f} m/s", end="     ")
        sys.stdout.flush()
    
    def print_instructions(self):
        """Print the keyboard control instructions"""
        os.system('clear')
        print("QCar2 Manual Control")
        print("====================")
        print("Use the following keys to control the QCar2:")
        print("  Arrow Up:     Increase throttle")
        print("  Arrow Down:   Decrease throttle / Reverse")
        print("  Arrow Left:   Turn left")
        print("  Arrow Right:  Turn right")
        print("  Space:        Stop")
        print("  q or ESC:     Quit")
        print("\nCurrent control values:")
        print("----------------------------------------------")

def get_key(settings):
    """Get a single keypress from the user"""
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    
    if key == '\x1b':  # Arrow keys have an escape prefix
        try:
            key = sys.stdin.read(2)
            return {'[A': 'UP', '[B': 'DOWN', '[C': 'RIGHT', '[D': 'LEFT'}.get(key, 'ESC')
        except:
            return 'ESC'
    return key

def main():
    # Initialize ROS node
    rclpy.init()
    node = QCar2ManualDrive()
    
    # Create a thread for ROS spinning
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node))
    spin_thread.daemon = True
    spin_thread.start()
    
    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        # Process keyboard input in the main thread
        while rclpy.ok():
            key = get_key(old_settings)
            
            if key == 'q' or key == 'ESC':
                break
            elif key == 'UP':
                node.motor_throttle = min(node.max_throttle, 
                                         node.motor_throttle + node.throttle_increment)
            elif key == 'DOWN':
                node.motor_throttle = max(-node.max_throttle, 
                                         node.motor_throttle - node.throttle_increment)
            elif key == 'LEFT':
                node.steering_angle = min(node.max_steering, 
                                         node.steering_angle + node.steering_increment)
            elif key == 'RIGHT':
                node.steering_angle = max(-node.max_steering, 
                                         node.steering_angle - node.steering_increment)
            elif key == ' ':  # Space key
                node.motor_throttle = 0.0
                node.steering_angle = 0.0
            
            # Small delay to avoid CPU hogging
            time.sleep(0.05)
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        # Clean up
        rclpy.shutdown()
        print("\nManual control terminated.")

if __name__ == '__main__':
    main()