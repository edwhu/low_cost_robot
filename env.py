# A gymnasium environment for the Koch robot.
import gymnasium
from robot import Robot
from dynamixel import Dynamixel
import numpy as np

class KochRobotEnv(gymnasium.Env):
    min_joint_pos = 0
    max_joint_pos = 4095
    
    def __init__(self, device_name, disable_torque_on_close=True):
        follower_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name=device_name).instantiate()
        self.robot = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
        self.robot.name = 'follower'
        self.disable_torque_on_close = disable_torque_on_close
        self.reset_joint_positions = [2069, 1544, 1111, 2311, 2075, 2691]
        # action space is an array of 6 elements, each element is between -1 and 1  and continuous.
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32) 
        # observation space is a dictionary with a single key 'joints' and value is an array of 6 elements, each element is between -1 and 1 and continuous.
        self.observation_space = gymnasium.spaces.Dict({
            'joints': gymnasium.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.int32)
        })

    def step(self, action):
        assert action.shape == (6,)
        assert -1 <= action.min() and action.max() <= 1
        # apply action
        goal_pos = denormalize_array(action, self.min_joint_pos, self.max_joint_pos, np.int32)
        self.robot.set_goal_pos(goal_pos)

        # get observation dictionary
        obs = self._get_obs_dict()
        reward = 0
        terminated = False 
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs_dict(self):
        joint_pos = np.asarray(self.robot.read_position())
        obs_dict = {
            'joints': normalize_array(joint_pos, self.min_joint_pos, self.max_joint_pos, np.float32)
        }
        return obs_dict
    
    def reset(self):
        self.robot.set_goal_pos(self.reset_joint_positions)
        obs = np.asarray(self.robot.read_position())
        info = {}
        return obs, info


    def render(self):
        pass

    def close(self):
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
    follower_device_name = '/dev/tty.usbmodem58760435361'
    leader_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name='/dev/tty.usbmodem58760428591').instantiate()
    leader = Robot(leader_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
    leader.name = 'leader'
    leader.set_trigger_torque()

    env = KochRobotEnv(device_name=follower_device_name, disable_torque_on_close=True)
    env.reset()
    counter = 0
    for i in range(10000):
        leader_pos = np.asarray(leader.read_position(), dtype=np.int32)
        action = normalize_array(leader_pos, env.min_joint_pos, env.max_joint_pos, np.float32)
        obs, *_ = env.step(action)
        if counter % 5000 == 0:
            print('L', leader_pos)
            print('F', denormalize_array(obs['joints'], env.min_joint_pos, env.max_joint_pos, np.int32) )
            print('')
        counter += 1
    env.close()