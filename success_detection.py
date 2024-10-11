from enum import Enum
import numpy as np
import cv2

from cameras.opencv import OpenCVCamera, OpenCVCameraConfig

class BlockColor(Enum):
    RED = 'red'
    ORANGE = 'orange'
    YELLOW = 'yellow'
    GREEN = 'green'
    LIGHT_BLUE = 'light_blue'
    DARK_BLUE = 'dark_blue'
    PURPLE = 'purple'

# TODO: Ed: i think we can just directly assign enums IDs, like RED='red', 0
block_color_to_id_map = {x: i for i, x in enumerate(BlockColor)}

class CamLocation(Enum):
    HAND = 'hand'
    LEFT = 'left'
    RIGHT = 'right'
    UNDER = 'under'
    THIRD_PERSON = 'third_person'

class CameraParameters:
    def __init__(self, id, resolution, obs_min_max_crop=None, reward_min_max_crop=None, grasp_min_max_crop=None):
        self.id = id
        self.left_id = f'{id}_left'
        self.right_id = f'{id}_right'
        self.resolution = resolution
        self.obs_min_max_crop = ((0,0), resolution) if obs_min_max_crop is None else obs_min_max_crop
        self.reward_min_max_crop = ((0,0), resolution) if reward_min_max_crop is None else reward_min_max_crop
        self.grasp_min_max_crop = ((0,0), resolution) if grasp_min_max_crop is None else grasp_min_max_crop


# Color tuning
camera_parameters = {
    CamLocation.HAND: CameraParameters('15512737', (720, 1280), ((0,410), (720,1280)), ((417,700), (700, 1100)), ((417,700), (700, 1100))),
    CamLocation.UNDER: CameraParameters('23007103', (720, 1280), None, None, None),
    CamLocation.THIRD_PERSON: CameraParameters('26368109', (720, 1280), ((160,451), (472,903)), ((255, 470), (410, 900)), None)  # TODO(js): Input crop region
}

hsv_color_ranges = {
    CamLocation.HAND: {  # color: (low, high)
        BlockColor.RED: (np.array([0, 50, 50,], dtype=np.uint8), np.array([6, 255, 255,], dtype=np.uint8)),
        BlockColor.ORANGE: (np.array([8, 50, 120,], dtype=np.uint8), np.array([14, 255, 255,], dtype=np.uint8)),
        BlockColor.YELLOW: (np.array([17, 150, 150,], dtype=np.uint8), np.array([27, 255, 255,], dtype=np.uint8)),
        BlockColor.GREEN: (np.array([62, 50, 50,], dtype=np.uint8), np.array([79, 255, 255,], dtype=np.uint8)),
        BlockColor.LIGHT_BLUE: (np.array([79, 79, 79,], dtype=np.uint8), np.array([113, 255, 255], dtype=np.uint8)),
        BlockColor.DARK_BLUE: (np.array([115, 50, 15,], dtype=np.uint8), np.array([135, 255, 100,], dtype=np.uint8)),
        BlockColor.PURPLE: (np.array([155, 50, 15,], dtype=np.uint8), np.array([175, 255, 255,], dtype=np.uint8)),
    },
    CamLocation.UNDER: {  # color: (low, high)
        BlockColor.RED: (np.array([0, 50, 50,], dtype=np.uint8), np.array([7, 255, 255,], dtype=np.uint8)),
        BlockColor.ORANGE: (np.array([9, 100, 50,], dtype=np.uint8), np.array([17, 255, 255,], dtype=np.uint8)),
        BlockColor.YELLOW: (np.array([17, 75, 50,], dtype=np.uint8), np.array([27, 255, 255,], dtype=np.uint8)),
        BlockColor.GREEN: (np.array([38, 23, 31,], dtype=np.uint8), np.array([70, 255, 213,], dtype=np.uint8)),
        BlockColor.LIGHT_BLUE: (np.array([98, 50, 50,], dtype=np.uint8), np.array([110, 255, 255,], dtype=np.uint8)),
        BlockColor.DARK_BLUE: (np.array([112, 50, 25,], dtype=np.uint8), np.array([130, 255, 255,], dtype=np.uint8)),
        BlockColor.PURPLE: (np.array([146, 30, 30,], dtype=np.uint8), np.array([165, 255, 255,], dtype=np.uint8)),
    },
    CamLocation.THIRD_PERSON: {  # color: (low, high)
        BlockColor.RED: (np.array([0, 50, 50,], dtype=np.uint8), np.array([6, 255, 255,], dtype=np.uint8)),
        BlockColor.ORANGE: (np.array([9, 50, 120,], dtype=np.uint8), np.array([14, 255, 255,], dtype=np.uint8)),
        BlockColor.YELLOW: (np.array([17, 100, 100,], dtype=np.uint8), np.array([27, 255, 255,], dtype=np.uint8)),
        BlockColor.GREEN: (np.array([62, 50, 50,], dtype=np.uint8), np.array([79, 255, 255,], dtype=np.uint8)),
        BlockColor.LIGHT_BLUE: (np.array([100, 50, 50,], dtype=np.uint8), np.array([111, 255, 180,], dtype=np.uint8)),
        BlockColor.DARK_BLUE: (np.array([112, 50, 15,], dtype=np.uint8), np.array([135, 255, 100,], dtype=np.uint8)),
        BlockColor.PURPLE: (np.array([155, 50, 15,], dtype=np.uint8), np.array([175, 255, 255,], dtype=np.uint8)),
    },
}

id_to_cam_loc = { cam_params.id: cam_loc for cam_loc, cam_params in camera_parameters.items() }

def gui_color_threshold(image_path):
    # ------ GUI for determining color threshold ------
    def nothing(x):
        pass
    # Create a window
    cv2.namedWindow('image')

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv

    (hMin, sMin, vMin), (hMax, sMax, vMax) = hsv_color_ranges[CamLocation.HAND][BlockColor.LIGHT_BLUE]
    (phMin, psMin, pvMin), (phMax, psMax, pvMax) = hsv_color_ranges[CamLocation.HAND][BlockColor.LIGHT_BLUE]

    cv2.createTrackbar('HMin', 'image', hMin, 179, nothing)
    cv2.createTrackbar('SMin', 'image', sMin, 255, nothing)
    cv2.createTrackbar('VMin', 'image', vMin, 255, nothing)
    cv2.createTrackbar('HMax', 'image', hMax, 179, nothing)
    cv2.createTrackbar('SMax', 'image', sMax, 255, nothing)
    cv2.createTrackbar('VMax', 'image', vMax, 255, nothing)

    # Set default value for Max HSV trackbars
    # cv2.setTrackbarPos('HMax', 'image', 179)
    # cv2.setTrackbarPos('SMax', 'image', 255)
    # cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize HSV min/max values

    # hMin = sMin = vMin = hMax = sMax = vMax = 
    # phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



def streaming_gui_color_threshold(camera):
    # ------ GUI for determining color threshold ------
    def nothing(x):
        pass
    # Create a window
    cv2.namedWindow('image')

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv

    (hMin, sMin, vMin), (hMax, sMax, vMax) = hsv_color_ranges[CamLocation.HAND][BlockColor.LIGHT_BLUE]
    (phMin, psMin, pvMin), (phMax, psMax, pvMax) = hsv_color_ranges[CamLocation.HAND][BlockColor.LIGHT_BLUE]

    cv2.createTrackbar('HMin', 'image', hMin, 179, nothing)
    cv2.createTrackbar('SMin', 'image', sMin, 255, nothing)
    cv2.createTrackbar('VMin', 'image', vMin, 255, nothing)
    cv2.createTrackbar('HMax', 'image', hMax, 179, nothing)
    cv2.createTrackbar('SMax', 'image', sMax, 255, nothing)
    cv2.createTrackbar('VMax', 'image', vMax, 255, nothing)

    # Set default value for Max HSV trackbars
    # cv2.setTrackbarPos('HMax', 'image', 179)
    # cv2.setTrackbarPos('SMax', 'image', 255)
    # cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize HSV min/max values

    # hMin = sMin = vMin = hMax = sMax = vMax = 
    # phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        image = camera.async_read()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def color_threshold(image, cam_loc, color):
    # Convert to HSV format
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Get the color range
    (hMin, sMin, vMin), (hMax, sMax, vMax) = hsv_color_ranges[cam_loc][color]
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    # Color threshold
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    return mask, result


if __name__ == "__main__":
    # Load image
    # image = cv2.imread(image_path)

    # Use webcam
    side_camera_config = OpenCVCameraConfig(fps=30, width=640, height=480, color_mode='bgr')
    side_camera = OpenCVCamera(camera_index=1, config=side_camera_config)
    
    side_camera.connect()

    # streaming_gui_color_threshold(side_camera)

    # image = side_camera.async_read()
    # crop the image to just the bottom half, and left half
    # image = image[240:, :320]
    # gui_color_threshold(image)

    while True: 
        image = side_camera.async_read()
        crop_image = image[240:, :320]
        mask, mask_img = color_threshold(crop_image.copy(), CamLocation.HAND, BlockColor.LIGHT_BLUE)
        cv2.imshow('mask', mask_img)
        # show both the mask_img and the original image
        both_img = np.hstack((mask_img, crop_image))
        cv2.imshow('both', both_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    side_camera.disconnect()