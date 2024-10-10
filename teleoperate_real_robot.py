from robot import Robot
from dynamixel import Dynamixel

leader_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name='/dev/tty.usbmodem58760428591').instantiate()
# print('leader')
follower_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name='/dev/tty.usbmodem58760435361').instantiate()
# print('follower')
follower = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
leader = Robot(leader_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
leader.set_trigger_torque()
# print('set trigger torque')


counter = 0
while True:
    leader_pos = leader.read_position()
    follower.set_goal_pos(leader_pos)
    # if counter % 500 == 0:
    #     print('F', leader_pos)
    #     follower_pos = follower.read_position()
    #     print('L', follower_pos)
    #     print('')
    counter += 1