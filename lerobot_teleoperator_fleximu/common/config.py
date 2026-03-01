#Configuration

import numpy as np

#UDP config
HOST = '0.0.0.0'
IMU_PORT = 1399
FLEX_PORT = 1400

#Joint order
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift", 
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]

#IMU ID mapping
IMU_MAPPING = {
    '00010085': 'upperarm',
    '00010120': 'lowerarm',
    '00010143': 'hand'
}

#Mujoco joint ranges
MUJOCO_RANGES = {
    "shoulder_pan": (-1.9722220547535922, 1.9722220547535922),
    "shoulder_lift": (-1.8849555921538759, 1.7453292519943366),
    "elbow_flex": (-1.69, 1.5707963267948966),
    "wrist_flex": (-1.710422666954443, 1.710422666954443),
    "wrist_roll": (-2.775073510670984, 2.8623399732707),
    "gripper": (-0.17453297762778586, 1.7453291995659765)
}

MUJOCO_RANGES_EXPECTED = {
    "shoulder_pan": (-1.9722220547535922, 1.9722220547535922),
    "shoulder_lift": (-1.8849555921538759, 1.7453292519943366),
    "elbow_flex": (-1.69, 1.5707963267948966),
    "wrist_flex": (-1.710422666954443, 1.710422666954443),
    "wrist_roll": (-2.775073510670984, 2.8623399732707),
    "gripper": (-0.17453297762778586, 1.2)
}

ZERO_POSE = np.array([0.0, np.pi/2, -np.pi/2, 0.0, 0.0, 1.0])
HOME_POSE = np.array([0, -1.88, 1.56, 1.22, -1.44, 0])