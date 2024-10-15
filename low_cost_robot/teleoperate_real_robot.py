from robot import Robot
from dynamixel import Dynamixel

leader_dynamixel = Dynamixel.Config(baudrate=2_000_000, device_name='/dev/tty.usbmodem58760428591').instantiate()
# print('leader')
follower_dynamixel = Dynamixel.Config(baudrate=2_000_000, device_name='/dev/tty.usbmodem58760435361').instantiate()
# print('follower')
follower = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
follower.name = 'follower'
# follower._disable_torque()
leader = Robot(leader_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
leader.name = 'leader'
leader.set_trigger_torque()
# print('set trigger torque')


import threading
import time

follower_positions = []
stop_event = threading.Event()

read_rate_hz = 10  # Desired read rate in hertz
read_interval = 1.0 / read_rate_hz  # Calculate the interval in seconds

def read_follower_position():
    global follower_positions
    while not stop_event.is_set():
        follower_positions = follower.read_position()
        time.sleep(read_interval)  # Control the read rate

# Start the thread for reading follower position
# follower_thread = threading.Thread(target=read_follower_position)
# follower_thread.daemon = True
# follower_thread.start()

counter = 0
try: 
    while True:
        leader_pos = leader.read_position()
        follower.set_goal_pos(leader_pos)
        if counter % 10 == 0:
            pass
            follower_pos = follower.read_position()
            # print('F', follower_positions)
        #     print('F', leader_pos)
            # follower_pos = follower.read_position()
            print('F', follower_pos)
        #     print('')
        counter += 1
except KeyboardInterrupt: 
    # Set the event to stop the thread
    stop_event.set()
    # Wait for the thread to finish
    follower_thread.join()
    print("Follower thread terminated safely.")