import gym
import numpy as np

class CarlaObservations():

    def __init__(self, img_height, img_width, obs_sensor):

        self.img_height = img_height
        self.img_width = img_width
        self.obs_sensor = obs_sensor
        self._bev_wrapper = None

    def get_observation_space(self):
        if 'birdview' in self.obs_sensor:
            return gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(low=0, high=255, shape=(84, 84, 5), dtype=np.uint8),
                    "speed": gym.spaces.Box(0., 1.0, shape=(1, ))
                }
            )
        else:
            return gym.spaces.Box(low=0.0, high=255.0, shape=(self.img_height, self.img_width, 3), dtype=np.uint8)
