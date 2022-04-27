import math
import numpy as np



R = np.array([[ 7.07750e-01, -7.06402e-01,  9.26540e-03],
            [ 1.37921e-02,  7.03322e-04, -9.99905e-01],
            [ 7.06329e-01,  7.07810e-01,  1.02405e-02]])

def radian2angle(radian):
    return radian / math.pi * 180

def angle2radian(angle):
    return angle / 180 * math.pi

def roll(R):
    return math.atan2(R[1, 0], R[0, 0])

def yaw(R):
    return math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2))

def pitch(R):
    return math.atan2(R[2, 1], R[2, 2])

print(radian2angle(roll(R)), radian2angle(yaw(R)), radian2angle(pitch(R)))