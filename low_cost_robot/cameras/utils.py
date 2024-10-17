from pathlib import Path
from typing import Protocol

import cv2
import einops
import numpy as np

from datetime import datetime, timezone

import platform
import time


def busy_wait(seconds):
    if platform.system() == "Darwin":
        # On Mac, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)


class RobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(
        self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class RobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)

def capture_timestamp_utc():
    return datetime.now(timezone.utc)



def write_shape_on_image_inplace(image):
    height, width = image.shape[:2]
    text = f"Width: {width} Height: {height}"

    # Define the font, scale, color, and thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)  # Blue in BGR
    thickness = 2

    position = (10, height - 10)  # 10 pixels from the bottom-left corner
    cv2.putText(image, text, position, font, font_scale, color, thickness)


def save_color_image(image, path, write_shape=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_shape:
        write_shape_on_image_inplace(image)
    cv2.imwrite(str(path), image)


def save_depth_image(depth, path, write_shape=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

    if write_shape:
        write_shape_on_image_inplace(depth_image)
    cv2.imwrite(str(path), depth_image)


def convert_torch_image_to_cv2(tensor, rgb_to_bgr=True):
    assert tensor.ndim == 3
    c, h, w = tensor.shape
    assert c < h and c < w
    color_image = einops.rearrange(tensor, "c h w -> h w c").numpy()
    if rgb_to_bgr:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    return color_image


# Defines a camera type
class Camera(Protocol):
    def connect(self): ...
    def read(self, temporary_color: str= None) -> np.ndarray: ...
    def async_read(self) -> np.ndarray: ...
    def disconnect(self): ...
