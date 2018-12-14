"""Module for connecting to Sphero and using data from API to control it"""

from spheropy.Sphero import Sphero
import numpy as np
import json
from time import sleep
import random
from pprint import pprint as pp

class NeuroSphero:
    """
    A class that receives data from Neurosteers's Api, sets up connection to the Sphero ball
    and gives orders to the ball upon processed data.

    Args:
        sphero id: user's sphero
        features: the biomarkers on which the user want perform analysis
     """

    def __init__(self, sphero_id):
        self.sphero_ball = Sphero("NAME", sphero_id, response_time_out=2, number_tries=5)
        self.buf_size = 10
        self.buf = np.zer
        self.sample_number = 0
        self.y_prediction = [0.0, 0.0, 0.0, 0.0]
        return

    def connect(self):
        try:
            print("connecting to sphero ball...")
            self.sphero_ball.connect()
            for i in range(5):
                self.sphero_ball.set_color(255, 255, 0)  #yello
                sleep(0.5)
                self.sphero_ball.set_color(148, 0, 211)  #purple
                sleep(0.5)
                self.sphero_ball.set_color(255, 0, 0)    #red
                sleep(0.5)
            if self.sphero_ball.ping()[0]:
                print("sphero ball connected!")
                self.sphero_ball.set_color(0, 255, 0)
        except ValueError:
            print("Could not connect to sphero ball")
            print("please make sure sphero is on and bluetooth is on")
            return False

        return True

    def make_a_step(self, current_angle, speed, sleep_time):
        self.sphero_ball.roll(speed, current_angle)
        sleep(sleep_time)
        self.sphero_ball.roll(0, current_angle)

    def make_a_circle(self, steps=10):
        speed = 0x30
        sleep_time = 0.3
        rotate_by = 360 // steps
        current_angle = 1
        for _ in range(steps):
            self.make_a_step(current_angle % 360, speed, sleep_time)
            current_angle += rotate_by

    def blink(self, blink_rate=1):
        self.sphero_ball.set_inactivity_timeout(3600)

        for _ in range(5):
            blink_rate = abs(blink_rate - 0.05)
            self.sphero_ball.set_color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            sleep(blink_rate)

    def make_a_square(self):
        speed = 0x88
        sleep_time = 1
        for angle in [1, 90, 180, 270]:
            self.sphero_ball.roll(speed, angle)
            sleep(sleep_time)
        self.sphero_ball.roll(0, 0)




    def control_sphero(self, features):
        pass


