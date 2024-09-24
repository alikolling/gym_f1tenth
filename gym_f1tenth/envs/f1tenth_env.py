import time

import cv2
import gymnasium as gym
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from gymnasium import spaces
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

STATE_H, STATE_W = 100, 100


class F1tenthEnv(gym.Env, Node):

    def __init__(self):
        super(F1tenthEnv, self).__init__("gym_f1tenth")

        self.pub_cmd_steering = self.create_publisher(
            Float32, "/autodrive/f1tenth_1/steering_command", 1
        )
        self.pub_cmd_throttle = self.create_publisher(
            Float32, "/autodrive/f1tenth_1/throttle_command", 1
        )
        self.last_action = [0.0, 0.0]

        self.min_range = 0.25
        self.max_range = 8.0

        self.img_rows = STATE_H
        self.img_cols = STATE_W
        self.img_channels = 3

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.img_channels, self.img_rows, self.img_cols),
                    dtype=int,
                ),
                "scan": spaces.Box(
                    low=-(2**63), high=2**63 - 2, shape=(20,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_laser(self):
        data = None
        while data is None:
            try:
                data = rclpy.wait_for_message(
                    LaserScan, "/autodrive/f1tenth_1/lidar", time_to_wait=5
                )
            except:
                pass
        scan = np.asarray(data.ranges)
        scan[np.isnan(scan)] = self.min_range
        scan[np.isinf(scan)] = self.max_range

        return scan

    def _get_camera(self):
        image_data = None
        success = False
        cv_image = None
        while image_data is None:
            try:
                image_data = rclpy.wait_for_message(
                    Image, "/autodrive/f1tenth_1/front_camera", time_to_wait=5
                )
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                # temporal fix, check image is not corrupted

            except:
                pass
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Current_O_pixel", cv_image)
        # cv2.waitKey(3)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        cv_image = cv_image.reshape(
            self.img_channels, cv_image.shape[0], cv_image.shape[1]
        )
        return cv_image

    def _get_obs(self):
        return {"image": self._get_camera(), "scan": self._get_laser()}

    def _get_info(self):
        time_info = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - self.start_time)
        )
        time_info += "-" + str(self.num_timesteps)
        return {"time_info": time_info}

    def reset(self):

        throttle_command = Float32()
        throttle_command.data = 0.0
        self.pub_cmd_throttle.publish(throttle_command)

        steering_command = Float32()
        steering_command.data = 0.0
        self.pub_cmd_steering.publish(steering_command)

        observation = self._get_obs()

        return observation, self._get_info()

    def _get_reward(self, observation):
        reward = 0.0
        return reward

    def step(self, action):
        observation = self._get_obs()

        reward = self._get_reward(observation)

        terminated = False

        throttle_command = Float32()
        throttle_command.data = action[0]
        self.pub_cmd_throttle.publish(throttle_command)

        steering_command = Float32()
        steering_command.data = action[1]
        self.pub_cmd_steering.publish(steering_command)

        self.last_action = action

        return observation, reward, terminated, False, self._get_info()
