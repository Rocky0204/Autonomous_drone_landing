#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Empty
import time
import math

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_publisher = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.land_publisher = self.create_publisher(Empty, '/simple_drone/land', 10)

        # Parameters
        self.linear_velocity = 0.5  # m/s
        self.angular_velocity = 0.5  # rad/s
        self.altitude_velocity = 0.3  # m/s

        # Current Position Tracking
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_angle = 0.0  # Facing forward (0 radians)
        self.current_altitude = 0.0  # Simulated altitude tracking

    def move_to_target(self, target_x: float, target_y: float):
        """
        Moves drone to (target_x, target_y) with smooth altitude control.
        """
        # Takeoff and stabilize
        print("Taking off...")
        self.takeoff_publisher.publish(Empty())
        time.sleep(3)  # Give time to take off

        # Smooth altitude increase
        self.change_altitude(4.5)
        time.sleep(5)  

        # Compute required movement
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)

        # Rotate towards target
        angle_to_rotate = target_angle - self.current_angle
        angle_to_rotate = (angle_to_rotate + math.pi) % (2 * math.pi) - math.pi  # Normalize
        self.rotate(angle_to_rotate)

        # Move forward
        self.move_forward(distance)
        time.sleep(5)  # Wait before landing

        # Smooth landing
        self.land_smoothly()

        # Update current position
        self.current_x = target_x
        self.current_y = target_y
        self.current_angle = target_angle

    def move_forward(self, distance: float):
        """
        Moves the drone forward by a specified distance.
        """
        duration = distance / self.linear_velocity
        self.publish_cmd_vel(Vector3(x=self.linear_velocity))
        time.sleep(duration)
        self.publish_cmd_vel()  # Stop

    def rotate(self, angle: float):
        """
        Rotates the drone by the specified angle (radians).
        """
        duration = abs(angle) / self.angular_velocity
        self.publish_cmd_vel(angular_vec=Vector3(z=self.angular_velocity if angle > 0 else -self.angular_velocity))
        time.sleep(duration)
        self.publish_cmd_vel()  # Stop rotation

    def change_altitude(self, target_altitude: float):
        """
        Adjusts the altitude smoothly instead of making a sudden jump.
        """
        print(f"Adjusting altitude to {target_altitude} meters...")

        while abs(self.current_altitude - target_altitude) > 0.1:  # Allow minor error margin
            if self.current_altitude < target_altitude:
                self.publish_cmd_vel(Vector3(z=self.altitude_velocity))
                self.current_altitude += self.altitude_velocity * 0.5  # Simulating altitude increase
            else:
                self.publish_cmd_vel(Vector3(z=-self.altitude_velocity))
                self.current_altitude -= self.altitude_velocity * 0.5  # Simulating altitude decrease
            time.sleep(0.5)  # Small step adjustments

        self.publish_cmd_vel()  # Stop vertical movement
        print(f"Reached altitude: {self.current_altitude}m")

    def land_smoothly(self):
        """
        Lands the drone smoothly until it reaches the ground.
        """
        print("Landing...")

        while self.current_altitude > 0.1:  # Continue descending until close to ground
            self.publish_cmd_vel(Vector3(z=-self.altitude_velocity))
            self.current_altitude -= self.altitude_velocity * 0.5  # Simulating descent
            time.sleep(0.5)

        # Final soft landing
        self.publish_cmd_vel(Vector3(z=-0.1))  # Very slow descent near ground
        time.sleep(2)

        # Stop all motion
        self.publish_cmd_vel()
        print("Landed successfully.")

        # Send land message (optional)
        self.land_publisher.publish(Empty())

    def publish_cmd_vel(self, linear_vec=Vector3(), angular_vec=Vector3()):
        """
        Publishes a Twist message to control the drone.
        """
        twist = Twist(linear=linear_vec, angular=angular_vec)
        self.cmd_vel_publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    teleop_node = TeleopNode()

    while True:
        try:
            target_x = float(input("Enter target x coordinate: "))
            target_y = float(input("Enter target y coordinate: "))
            teleop_node.move_to_target(target_x, target_y)

            choice = input("Do you want to enter new coordinates? (yes/no): ").strip().lower()
            if choice != 'yes':
                break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except ValueError:
            print("Invalid input. Please enter numeric values.")

    teleop_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
