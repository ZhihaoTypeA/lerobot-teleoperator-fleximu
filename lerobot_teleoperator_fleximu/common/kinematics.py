import numpy as np
import time
from scipy.spatial.transform import Rotation as R

class GeometricSolver:
    def __init__(self):
        self.HUMAN_UPPER_LEN = 0.34
        self.HUMAN_LOWER_LEN = 0.28
        self.ROBOT_LINK1 = 0.116
        self.ROBOT_LINK2 = 0.135
        self.ua_base_offset = np.array([0.0692345, 0.0, 0.1166])
        self.scale_factor = (self.ROBOT_LINK1 + self.ROBOT_LINK2) / (self.HUMAN_UPPER_LEN + self.HUMAN_LOWER_LEN)

    def human_fk(self, q_upper, q_lower):
        r_up = R.from_quat(q_upper)
        r_lo = R.from_quat(q_lower)
        vec_upper = np.array([self.HUMAN_UPPER_LEN, 0, 0])
        vec_lower = np.array([self.HUMAN_LOWER_LEN, 0, 0])
        elbow_pos = r_up.apply(vec_upper)
        wrist_pos = elbow_pos + r_lo.apply(vec_lower)
        return wrist_pos

    def robot_ik(self, target_pos, hand_quat):
        x, y, z = target_pos - self.ua_base_offset
        L1 = self.ROBOT_LINK1
        L2 = self.ROBOT_LINK2
        
        theta1 = np.arctan2(y, x)
        
        r_proj = np.sqrt(x**2 + y**2)
        h = z
        dist_sq = r_proj**2 + h**2
        max_reach = L1 + L2
        
        if dist_sq > max_reach**2:
            scale = max_reach / np.sqrt(dist_sq)
            r_proj *= scale
            h *= scale
            dist_sq = max_reach**2
        
        cos_theta3 = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        theta3 = np.arccos(cos_theta3)
        
        alpha = np.arctan2(h, r_proj)
        cos_beta = (dist_sq + L1**2 - L2**2) / (2 * L1 * np.sqrt(dist_sq))
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = np.arccos(cos_beta)
        
        theta2 = alpha + beta 
        
        r_hand = R.from_quat(hand_quat)
        _, pitch_global, _ = r_hand.as_euler('ZYX', degrees=False)
        theta4 = - pitch_global - (theta2 - theta3)
        
        _, _, roll_global = r_hand.as_euler('ZYX', degrees=False)
        theta5 = roll_global

        return np.array([theta1, theta2-np.pi/2, -theta3+np.pi/2, theta4, theta5])

    def robot_fk(self, q1, q2, q3, auto_invert=True):
        L1 = self.ROBOT_LINK1
        L2 = self.ROBOT_LINK2

        if auto_invert:
            q1, q2, q3 = -q1, -q2, -q3

        theta1 = q1
        theta2 = q2 + np.pi / 2.0
        theta3 = np.pi / 2.0 - q3

        r_proj = L1 * np.cos(theta2) + L2 * np.cos(theta2 - theta3)
        h = L1 * np.sin(theta2) + L2 * np.sin(theta2 - theta3)

        x = r_proj * np.cos(theta1)
        y = r_proj * np.sin(theta1)
        z = h

        return np.array([x, y, z]) + self.ua_base_offset
