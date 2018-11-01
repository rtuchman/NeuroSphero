import NeuroStream
from spheropy.Sphero import Sphero
import numpy
import json


class NeuroSphero:
    # connections
    nstream = ''
    sp = ''
    # parameters
    calibration_samples = 0
    features = ['','']

    # mode decision
    state = 'training'
    nsamples = 0

    #calibration data
    buf = []

    # running calculated parameters
    std = 0
    mean = 0
    max  = 0
    min = 0

    def __init__(self, calibration_samples, features):
        self.calibration_samples = calibration_samples
        self.buf = numpy.zeros([len(features), self.calibration_samples])
        self.features = features
        return

    def __enter__(self):
        self.nstream.__enter__()
        self.sp.__enter__()
        print('connected!')
        return self

    def __exit__(self, type, value, traceback):
        return

    @staticmethod
    def connect(email, password, sensor, sphero_address, calibration_samples = 30, features = ['h1']):
        neurosphero = NeuroSphero(calibration_samples, features)
        neurosphero.nstream = NeuroStream.NeuroStream.connect(email = email, password = password, sensor = sensor)
        neurosphero.sp = Sphero("NAME", sphero_address, response_time_out=2, number_tries=10)
        return neurosphero

    def stream(self):
        self.nstream.stream(self.on_message, self.on_error)

    def perform_calibration(self, features):
        self.sp.set_color(255, 255, 255)
        print('calibrating sample ' + '%d' % self.nsamples)
        for buf, feature in zip(self.buf, self.features):
            buf[self.nsamples - 1]=features[feature]

        if self.nsamples == self.calibration_samples:
            self.state = 'training'
            self.mean = [(2*numpy.mean(buf) + 0.2) / 3 for buf in self.buf]
            self.std = [(2*numpy.std(buf)+0.1)/3  for buf in self.buf]
            self.min = [(2*numpy.min(buf) + 0) / 3 for buf in self.buf]
            self.max = [(2*numpy.max(buf) + 0.4) / 3 for buf in self.buf]

    def control_sphero(self, features):
        up_thresh = [min(mean + 2 * std_val, max_val) for mean, std_val, max_val in zip(self.mean, self.std, self.max)]
        down_thresh = [max(mean - 2 * std_val, min_val) for mean, std_val, min_val in
                       zip(self.mean, self.std, self.min)]

        str = ''
        for i in range(len(self.features)):
            str += (self.features[i] + ': ' + '%.2f' % ((features[self.features[i]] + 1) * 50) + 'up_thresh: ' + '%.2f' % (
            (up_thresh[i] + 1) * 50) + 'down_thresh: ' + '%.2f' % ((down_thresh[i] + 1) * 50) + '\t')
        print(str)

        if len(self.features) >= 1:
            if features[self.features[0]] > up_thresh[0]:
                self.sp.roll(30, 90)
            elif features[self.features[0]] < down_thresh[0]:
                self.sp.roll(30, 270)
        if len(self.features) >= 2:
            if features[self.features[1]] > up_thresh[1]:
                self.sp.set_color(0, 255, 0)
            elif features[self.features[1]] < down_thresh[1]:
                self.sp.set_color(0, 0, 255)
            else:
                self.sp.set_color(100, 100, 100)

    def on_message(self, ws, message):
        # print message
        message = json.loads(message)
        features = message[u'features']
        qf = features[u'qf']

        # check if data is valid
        if qf != 0:
            self.sp.set_color(255, 0, 0)
            return

        # training mode
        self.nsamples = min(self.nsamples+1,1000)
        if self.nsamples <= self.calibration_samples:
            self.perform_calibration(features)
            return


        # controling mode
        self.control_sphero(features)



        c1 = features[u'c1']
        h1 = features[u'h1']
        speed = 0

    def on_error(self, ws, error):
        print("sphero_err" + error)


if __name__ == "__main__":
    with NeuroSphero.connect(email='mymail', password='mypassword', sensor='sensorid', sphero_address="68:86:e7:04:4d:10", features = ['c4', 'h1']) as ns:
        ns.stream()
