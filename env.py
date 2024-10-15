# A gymnasium environment for the Koch robot.
import cv2
import gymnasium
from cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from robot import Robot
from dynamixel import Dynamixel
import numpy as np
from collections import defaultdict, deque
import sys
from tqdm import tqdm, trange
import time
import os

from success_detection import BlockColor, CamLocation, color_threshold

class KochRobotEnv(gymnasium.Env):
    min_joint_pos = 0
    max_joint_pos = 4095
    
    def __init__(self, device_name, cameras = None, disable_torque_on_close=True):
        follower_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name=device_name).instantiate()
        self.robot = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
        self.robot.name = 'follower'
        self.disable_torque_on_close = disable_torque_on_close
        self.reset_joint_positions = [2000, 1548, 1095, 2111, 2056, 2690]
        self.action_space = gymnasium.spaces.Box(low=0, high=4095, shape=(6,), dtype=np.int32) 
        _observation_space = {
            'joints': gymnasium.spaces.Box(low=0, high=4096, shape=(6,), dtype=np.int32),
            'gripper_stuck': gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
        }
        self.achieved_gripper_pos_deque = deque(maxlen=10)
        self.commanded_gripper_pos_deque = deque(maxlen=10)
        self.cameras = cameras
        if self.cameras is not None:
            # pass
            # assert len(cameras) == 2, "Assume we have wrist and side camera"
            # just give the wrist camera to the policy.
            _observation_space["wrist_cam"] = gymnasium.spaces.Box(low=0, high=255, shape=(640, 480, 3), dtype=np.uint8)
            # for camera in self.cameras.values():
            #     camera.connect()

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
        gripper_stuck = obs['gripper_stuck'][0]
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
        info['rew_img'] = crop_image

        reward = 0.5 * gripper_stuck +  (gripper_stuck) * (1.0 * block_detected)
        terminated = False 
        truncated = False
        return obs, reward, terminated, truncated, info

    def _get_obs_dict(self, action=None):
        joint_pos = np.asarray(self.robot.read_position())
        obs_dict = {
            'joints': joint_pos
        }
        if self.cameras is not None:
            # pass
            img = self.cameras["wrist_cam"].async_read()
            # crop out the top half of the image
            img = img[240:, :]
            resized_img = img
            obs_dict[f'wrist_cam'] = resized_img
        
        if action is None:
            obs_dict['gripper_stuck'] = np.array([0], dtype=np.int32)
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
            gripper_stuck = gripper_action_closing and gripper_in_range and gripper_unchanging
            # gripper_stuck and print('gripper stuck')
            obs_dict['gripper_stuck'] = np.array([gripper_stuck], dtype=np.int32)
        return obs_dict
    
    def reset(self):
        self.robot.set_goal_pos(self.reset_joint_positions)
        obs = self._get_obs_dict()
        info = {}
        return obs, info


    def render(self):
        pass

    def close(self):
        if self.cameras is not None:
            for camera in self.cameras.values():
                camera.disconnect()
        if self.disable_torque_on_close:
            self.robot._disable_torque()

    def seed(self):
        pass

def collect_demos(demo_folder):
    wrist_camera_config = OpenCVCameraConfig(fps=30, width=640, height=480, color_mode='rgb')
    wrist_camera = OpenCVCamera(camera_index=0, config=wrist_camera_config)
    wrist_camera.connect()

    side_camera_config = OpenCVCameraConfig(fps=30, width=640, height=480, color_mode='rgb')
    side_camera = OpenCVCamera(camera_index=1, config=side_camera_config)
    side_camera.connect()

    # print("sleeping for 2 seconds to let the cameras connect.")
    # time.sleep(2)

    cameras = {'side_cam': side_camera, 'wrist_cam': wrist_camera}

    follower_device_name = '/dev/tty.usbmodem58760435361'
    leader_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name='/dev/tty.usbmodem58760428591').instantiate()
    leader = Robot(leader_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
    leader.name = 'leader'
    leader.set_trigger_torque()

    env = KochRobotEnv(device_name=follower_device_name, cameras=cameras,disable_torque_on_close=True)

    demo_length = 350 # in steps
    reset_seconds = 5 # in seconds
    num_demos = 10
    demos_collected = 0

    while demos_collected < num_demos:
        ep_dict = defaultdict(list)
        obs, info = env.reset()
        for k, v in obs.items():
            ep_dict['obs/' + k].append(v)
        # Tell the user that the robot is ready for teleop, and wait for their input.
        input(f"Demo {demos_collected + 1}/10, Press Enter to start the collection.")
        for timestep in trange(demo_length, desc="Collecting demo"):
            leader_pos = np.asarray(leader.read_position(), dtype=np.int32)
            action = leader_pos
            ep_dict['action'].append(action)
            obs, rew, terminated, truncated, info = env.step(action)
            img = info['rew_img']
            # cv2.imshow('mask', info['mask'])
            cv2.imshow('side', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            img = obs['wrist_cam']
            cv2.imshow('wrist', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            if timestep % 10 == 0:
                print('L:', leader_pos, 'F:', obs['joints'], 'rew:', rew, 'mask_pixels:', info['mask_pixels'])
            timestep += 1
            for k, v in obs.items():
                ep_dict['obs/' + k].append(v)
            ep_dict['reward'].append(rew)
            ep_dict['terminated'].append(terminated)
            ep_dict['truncated'].append(truncated)
            for k, v in info.items():
                ep_dict['info/' + k].append(v)
        
        save_demo = input("Save the demo? enter y/n")
        if save_demo.lower() == 'y':
            demos_collected += 1
            demo_path = os.path.join(demo_folder, f'demo_{demos_collected}.npy')
            np.savez_compressed(demo_path, **ep_dict)
        else:
            print("Demo not saved.")
        
        print("Clean up the environment.")
        # Here, we would give the user some time to clean up the environment and the robot. 
        start_time = time.time()
        with tqdm(total=reset_seconds, desc="Waiting for reset...") as pbar:
            while time.time() - start_time < reset_seconds:
                leader_pos = np.asarray(leader.read_position(), dtype=np.int32)
                env.robot.set_goal_pos(leader_pos)
                pbar.update(time.time() - start_time)
        
    env.close()
        
if __name__ == '__main__':
    os.makedirs('demos', exist_ok=True)
    collect_demos('demos')
    sys.exit(0)


    """=======Code for testing out the robot========="""
    # wrist camera
    wrist_camera_config = OpenCVCameraConfig(fps=30, width=640, height=480, color_mode='rgb')
    wrist_camera = OpenCVCamera(camera_index=0, config=wrist_camera_config)
    wrist_camera.connect()

    side_camera_config = OpenCVCameraConfig(fps=30, width=640, height=480, color_mode='rgb')
    side_camera = OpenCVCamera(camera_index=1, config=side_camera_config)
    side_camera.connect()

    cameras = {
        'wrist_cam': wrist_camera, 
        'side_cam': side_camera}
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
        img = info['rew_img']
        cv2.imshow('mask', info['mask'])
        cv2.imshow('side', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        img = obs['wrist_cam']
        cv2.imshow('wrist', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        if counter % 1 == 0:
            print('L:', leader_pos, 'F:', obs['joints'], 'rew:', rew, 'mask_pixels:', info['mask_pixels'], end='\r')
        counter += 1
    env.close()
