"""Module for connecting to Sphero and using data from API to control it"""

from spheropy.Sphero import Sphero
import numpy
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

    def __init__(self, sphero_id, features=(u'c1', u'h1')):
        self.sphero_ball = Sphero("NAME", sphero_id, response_time_out=2, number_tries=5)
        self.calibration_samples = 30  # buffer size
        self.features = [x for x in features]
        self.buf = {feature: numpy.zeros([self.calibration_samples]) for feature in self.features}
        self.sample_number = 0
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

    def blink(self):
        self.sphero_ball.set_inactivity_timeout(3600)
        blink_rate = 1
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


    def perform_calibration(self, features):
        """
        Continuously saves the measured data into a buffer.
        Each feature (e.g c1 ,h1) is saved in it's own buffer.
        For each buffer it computes mean, std, min, max
        """
        buf_iter = (self.sample_number) % self.calibration_samples
        for feature in self.features:
            self.buf[feature][buf_iter] = features[feature]
        if self.sample_number <= self.calibration_samples:
            print('calibrating sample ' + '%d' % self.sample_number)
            pp(self.buf)
        if self.sample_number >= self.calibration_samples:  # after buffer is full (we have 30 or more samples).
            self.state = 'training'
            self.mean = {feature: (numpy.mean(self.buf[feature])) for feature in self.buf}
            self.std = {feature: (numpy.std(self.buf[feature])) for feature in self.buf}
            self.min = {feature: (numpy.min(self.buf[feature])) for feature in self.buf}
            self.max = {feature: (numpy.max(self.buf[feature])) for feature in self.buf}
            self.analytics = {'mean': self.mean, 'std': self.std, 'min': self.min, 'max': self.max}
            #pp(self.analytics)
        self.sample_number += 1

    def control_sphero(self, features):
        up_thresh = [min(mean + 2 * std_val, max_val) for mean, std_val, max_val
                     in zip(self.mean.values(), self.std.values(), self.max.values())]
        down_thresh = [max(mean - 2 * std_val, min_val) for mean, std_val, min_val in
                       zip(self.mean.values(), self.std.values(), self.max.values())]

        str_ = ''
        for i in range(len(self.features)):
            tmp = "{0}: {1:.2f}, up_thresh: {2:.2f}, down_thresh: {3:.2f}\n".format(self.features[i],
                                                                                    (features[self.features[i]] + 1)*50,
                                                                                    (up_thresh[i] + 1) * 50,
                                                                                    (down_thresh[i] + 1) * 50)
            str_ = ''.join([str_, tmp])
        flag = self.sample_number % 2
        print("\n{}".format(str_))
        if len(self.features) >= 1:
            if features[self.features[0]] > up_thresh[0]:  # if c1 > up_thresh: move forward
                self.sphero_ball.roll(30, 90 + 180 * flag)
            elif features[self.features[0]] < down_thresh[0]:  # if c1 < down_thresh: move backward
                self.sphero_ball.roll(30, 270)
        if len(self.features) >= 2:
            if features[self.features[1]] > up_thresh[1]:  # if h1 > up_thresh: green color
                self.sphero_ball.set_color(0, 255, 0)
            elif features[self.features[1]] < down_thresh[1]:  # if h1 < down_thresh: red color
                self.sphero_ball.set_color(255, 0, 0)
            else:
                self.sphero_ball.set_color(100, 100, 100)  # h1 is between thresholds: white color


