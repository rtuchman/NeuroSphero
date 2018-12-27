"""Module for connecting to Sphero and using data from API to control it"""

from spheropy.Sphero import Sphero
import numpy as np
import json
from time import sleep
import random
from threading import Thread

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
        self.sample_number = 1
        self.y_prediction = -1

        return

    def connect(self):
        try:
            print("connecting to sphero ball...")
            self.sphero_ball.connect()
            self.sphero_ball.set_color(255, 255, 255)
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
        speed = 0x20
        sleep_time = 0.3
        rotate_by = 360 // steps
        current_angle = 1
        for _ in range(3):  # 5 circles
            for _ in range(steps):
                self.make_a_step(current_angle % 360, speed, sleep_time)
                current_angle += rotate_by

    def blink(self, blink_rate=0.25):
        for _ in range(15):
            self.sphero_ball.set_color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            sleep(blink_rate)

    def make_a_square(self):
        speed = 0x30
        sleep_time = 1
        for _ in range(5):
            for angle in [1, 90, 180, 270]:
                self.sphero_ball.roll(speed, angle)
                sleep(sleep_time)
            self.sphero_ball.roll(0, 0)

    def control_sphero(self):
        while True:
            y = self.y_prediction  # pull prediction from main thread

            if y == 0:  # Memory game
                print('Memory game')
                for _ in range(10):
                    self.sphero_ball.set_color(0, 0, 255)
                    sleep(0.25)
                    self.sphero_ball.set_color(149, 0, 179)
                    sleep(0.25)


            if y == 1:  # Meditate
                print('Meditate')
                for _ in range(3):
                    for i in range(25, 260, 5):
                        self.sphero_ball.set_color(0, i, 0)
                        sleep(0.03)
                    for i in range(255, 25, -5):
                        self.sphero_ball.set_color(0, i, 0)
                        sleep(0.03)


            if y == 2:  # Write with weak hand
                print('Write with weak hand')
                for _ in range(19):
                    self.sphero_ball.set_color(255, 0, 255)
                    sleep(0.25)
                    self.sphero_ball.set_color(43, 0, 255)
                    sleep(0.25)


            if y == 3:  # Happy music (dancing)
                print('Happy music')
                self.thread_circle = Thread(target=self.make_a_circle)
                self.thread_blink = Thread(target=self.blink)
                self.thread_blink.start()
                self.thread_circle.start()
                self.thread_blink.join()
                self.thread_circle.join()

            if y == -1:  # No prediction
                for _ in range(19):
                    self.sphero_ball.set_color(255, 255, 255)
                    sleep(0.5)

            if y == -2:  # No prediction
                for _ in range(19):
                    self.sphero_ball.set_color(255, 0, 0)
                    sleep(0.5)












