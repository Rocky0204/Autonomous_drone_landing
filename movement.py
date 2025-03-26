#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import cv2
import numpy as np
from cv_bridge import CvBridge
import math
import time
from rclpy.duration import Duration


class RedMarkerLanding(Node):
    def __init__(self):
        super().__init__('red_marker_landing')

        # ROS2 Publishers & Subscribers
        self.image_sub = self.create_subscription(Image, '/simple_drone/bottom/image_raw', self.image_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.land_pub = self.create_publisher(Empty, '/simple_drone/land', 10)

        self.bridge = CvBridge()

        # Drone Status
        self.taken_off = False
        self.reached_altitude = False
        self.searching_marker = False
        self.altitude_step = 0

        # Square Spiral Search Parameters
        self.square_step = 0.5
        self.square_layer = 1
        self.square_moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.square_index = 0
        self.move_count = 0

        # Tolerances
        self.center_tolerance = 20
        self.altitude_velocity = 0.2

        self.get_logger().info("Taking off...")
        self.takeoff_pub.publish(Empty())
        self.altitude_timer = self.create_timer(1.0, self.ascend_to_altitude)

    def ascend_to_altitude(self):
        """ Ascend to a target altitude in steps. """
        if self.altitude_step < 10:  # Approximate 4m altitude
            twist = Twist()
            twist.linear.z = 0.3
            self.cmd_vel_pub.publish(twist)
            self.altitude_step += 1
        else:
            self.cmd_vel_pub.publish(Twist())  # Stop ascent
            self.reached_altitude = True
            self.get_logger().info("Reached altitude. Starting marker detection.")
            self.altitude_timer.cancel()
            self.start_square_spiral_search()

    def image_callback(self, msg):
        """ Process bottom camera image to detect red marker. """
        if not self.reached_altitude:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        height, width, _ = frame.shape

        # Convert to HSV and mask red color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            if self.searching_marker:
                self.spiral_timer.cancel()
                self.searching_marker = False
                self.get_logger().info("Marker found! Aligning...")

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            marker_center_x = x + w // 2
            marker_center_y = y + h // 2

            image_center_x = width // 2
            image_center_y = height // 2
            offset_x = marker_center_x - image_center_x
            offset_y = marker_center_y - image_center_y

            self.align_and_land(offset_x, offset_y)

    def start_square_spiral_search(self):
        """ Starts a square spiral search movement. """
        self.searching_marker = True
        self.spiral_timer = self.create_timer(0.5, self.square_spiral_move)
        self.get_logger().info("Marker not detected. Starting square spiral search...")

    def square_spiral_move(self):
        """ Moves the drone in a square spiral pattern to search for the marker. """
        if not self.searching_marker:
            return

        twist = Twist()
        dx, dy = self.square_moves[self.square_index]

        twist.linear.x = float(dx * self.square_step)
        twist.linear.y = float(dy * self.square_step)

        self.cmd_vel_pub.publish(twist)

        # Move to the next step in the spiral
        self.move_count += 1
        if self.move_count == self.square_layer:
            self.move_count = 0
            self.square_index = (self.square_index + 1) % 4

            if self.square_index == 0 or self.square_index == 2:
                self.square_layer += 1

    def align_and_land(self, offset_x, offset_y):
        """ Aligns and lands the drone when centered over the marker. """
        twist = Twist()
        move_x = move_y = False

        if abs(offset_x) > self.center_tolerance:
            twist.linear.y = -0.2 if offset_x > 0 else 0.2  # Move left/right
            move_x = True

        if abs(offset_y) > self.center_tolerance:
            twist.linear.x = -0.2 if offset_y > 0 else 0.2  # Move forward/backward
            move_y = True

        if move_x or move_y:
            self.get_logger().info(f"Aligning: offset_x={offset_x}, offset_y={offset_y}")
            self.cmd_vel_pub.publish(twist)
        else:
            self.smooth_landing()

    def smooth_landing(self):
        """ Gradually lands the drone smoothly. """
        self.get_logger().info("Marker centered. Initiating smooth landing...")

        if self.searching_marker:
            self.spiral_timer.cancel()
            self.searching_marker = False

        twist = Twist()
        descent_velocity = 0.2  # Start with a moderate descent speed

        for i in range(10):
            twist.linear.z = -descent_velocity
            self.cmd_vel_pub.publish(twist)
            rclpy.sleep(0.5)  # Wait to allow smooth descent
            descent_velocity *= 0.9  # Gradually decrease descent speed

        self.cmd_vel_pub.publish(Twist())  # Stop movement
        self.land_pub.publish(Empty())  # Send landing command
        self.get_logger().info("Drone landed successfully.")



def main(args=None):
    rclpy.init(args=args)
    node = RedMarkerLanding()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()