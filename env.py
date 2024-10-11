# A gymnasium environment for the Koch robot.
import gymnasium
from cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from robot import Robot
from dynamixel import Dynamixel
import numpy as np
from collections import deque

from success_detection import BlockColor, CamLocation, color_threshold

class KochRobotEnv(gymnasium.Env):
    min_joint_pos = 0
    max_joint_pos = 4095
    
    def __init__(self, device_name, cameras = None, disable_torque_on_close=True):
        follower_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name=device_name).instantiate()
        self.robot = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
        self.robot.name = 'follower'
        self.disable_torque_on_close = disable_torque_on_close
        self.reset_joint_positions = [2069, 1544, 1111, 2311, 2075, 2691]
        self.action_space = gymnasium.spaces.Box(low=0, high=4095, shape=(6,), dtype=np.int32) 
        _observation_space = {
            'joints': gymnasium.spaces.Box(low=0, high=4096, shape=(6,), dtype=np.int32),
            'object_detected': gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
        }
        self.achieved_gripper_pos_deque = deque(maxlen=10)
        self.commanded_gripper_pos_deque = deque(maxlen=10)
        self.cameras = cameras
        if self.cameras is not None:
            assert len(cameras) == 2, "Assume we have wrist and side camera"
            # just give the wrist camera to the policy.
            _observation_space["wrist_cam"] = gymnasium.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
            for camera in self.cameras.values():
                camera.connect()

        self.observation_space = gymnasium.spaces.Dict(_observation_space)

    def step(self, action):
        info = {}
        # apply action
        goal_pos = action
        self.robot.set_goal_pos(goal_pos)
        # get observation dictionary
        obs = self._get_obs_dict(action)
        
        # =======Reward=========
        # reward will be combination of object detected and if the block is in the target circle. 
        object_in_gripper = obs['object_in_gripper'][0]
        side_img = self.cameras['side_cam'].async_read()
        crop_image = side_img[320:, 120:320]
        # convert rgb to bgr
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)
        mask, _ = color_threshold(crop_image, CamLocation.HAND, BlockColor.LIGHT_BLUE)
        mask_pixels = mask.sum()
        block_detected = mask_pixels > 30000

        info['mask_pixels'] = mask_pixels
        info['block_detected'] = block_detected
        info['mask'] = mask

        reward = max(0.1 * object_in_gripper,  1.0 * block_detected)
        terminated = False 
        truncated = False
        return obs, reward, terminated, truncated, info

    def _get_obs_dict(self, action=None):
        joint_pos = np.asarray(self.robot.read_position())
        obs_dict = {
            'joints': joint_pos
        }
        if self.cameras is not None:
            img = self.cameras["wrist_cam"].async_read()
            # crop the image to the region of interest
            resized_img = cv2.resize(img, (128, 128))
            resized_img = img
            obs_dict[f'wrist_cam'] = resized_img
        
        if action is None:
            obs_dict['object_in_gripper'] = np.array([0], dtype=np.int32)
        else:
            # detect if object is gripped.
            commanded_gripper_action = action[-1]
            achieved_gripper_position = obs_dict['joints'][-1]

            self.achieved_gripper_pos_deque.append(achieved_gripper_position)
            self.commanded_gripper_pos_deque.append(commanded_gripper_action)
            gripper_action_closing = all([c < 2000 for c in self.commanded_gripper_pos_deque ])
            
            gripper_in_range = True
            # all([2350 < a < 2500 for a in self.achieved_gripper_pos_deque])
            gripper_unchanging = (np.abs(np.diff(self.achieved_gripper_pos_deque)) < 5).all()
            object_in_gripper = gripper_action_closing and gripper_in_range and gripper_unchanging
            # object_in_gripper and print('Object detected')
            obs_dict['object_in_gripper'] = np.array([object_in_gripper], dtype=np.int32)
        return obs_dict
    
    def reset(self):
        self.robot.set_goal_pos(self.reset_joint_positions)
        obs = np.asarray(self.robot.read_position())
        info = {}
        return obs, info


    def render(self):
        pass

    def close(self):
        if self.cameras is not None:
            for camera in self.cameras:
                camera.close()
        if self.disable_torque_on_close:
            self.robot._disable_torque()

    def seed(self):
        pass


def normalize_array(array: np.ndarray, min_value, max_value, dtype) -> np.ndarray:
    array = (array - min_value) / (max_value - min_value) # bound to [0, 1]
    return (array * 2 - 1).astype(dtype) # bound to [-1, 1]


def denormalize_array(array: np.ndarray, min_value, max_value, dtype) -> np.ndarray:
    array = (array + 1) / 2 # bound to [0, 1]
    return (array * (max_value - min_value) + min_value).astype(dtype) 

if __name__ == '__main__':
    import cv2

    # wrist camera
    wrist_camera_config = OpenCVCameraConfig(fps=30, width=640, height=480, color_mode='rgb')
    wrist_camera = OpenCVCamera(camera_index=0, config=wrist_camera_config)

    side_camera_config = OpenCVCameraConfig(fps=30, width=640, height=480, color_mode='rgb')
    side_camera = OpenCVCamera(camera_index=1, config=side_camera_config)

    cameras = {'wrist_cam': wrist_camera, 'side_cam': side_camera}
    # cameras = None

    follower_device_name = '/dev/tty.usbmodem58760435361'
    leader_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name='/dev/tty.usbmodem58760428591').instantiate()
    leader = Robot(leader_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
    leader.name = 'leader'
    leader.set_trigger_torque()

    env = KochRobotEnv(device_name=follower_device_name, cameras=cameras,disable_torque_on_close=True)
    env.reset()
    counter = 0
    for i in range(10000):
        leader_pos = np.asarray(leader.read_position(), dtype=np.int32)
        action = leader_pos
        obs, rew, terminated, truncated, info = env.step(action)
        img = obs['wrist_cam']
        # import ipdb; ipdb.set_trace()
        cv2.imshow('mask', info['mask'])
        cv2.imshow('wrist', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        if counter % 150 == 0:
            print('L:', leader_pos[-1], 'F:', obs['joints'][-1], 'rew:', rew, 'mask_pixels:', info['mask_pixels'])
        counter += 1
    env.close()
