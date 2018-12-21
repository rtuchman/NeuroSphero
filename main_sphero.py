import threading
import websocket
#import sys
#from functools import partial
#import time

from NeuroSphero import *
from NeuroLogin import *
from NeuroLogout import disconnect as disconnect_neuro
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# EMAIL = 'runtuchman@gmail.com'
# PASSWORD = '1234Ran'

EMAIL = 'matanron3@gmail.com'
PASSWORD = 'Matan1234'

SENSOR = '00a3b4810811' #'b827eb0b7120' # ''810811' # new
SPHERO_ID = '68:86:e7:01:fb:b2' #obr

# SENSOR =  '00a3b4d8a9a7' # old
# SPHERO_ID = '68:86:e7:04:4d:10' #ypr

class NeuroSpheroManager(object):
    """Neuro sphero manager in charge of managing the connections of neuro sensor and sphero balls."""
    def __init__(self, email=EMAIL, password=PASSWORD, sensor=SENSOR, sphero_id=SPHERO_ID):
        self.email = email
        self.password = password
        self.sensor = sensor
        self.sphero_id = sphero_id

        self.running = False  # indication whether we want to read data from the sensor or not
        self.ws = self.connect()

        self.is_training = False
        self.neurosphero.y_prediction = -1
        self.neurolearn = load_model('NeuroClassifier.h5')

        print('created neuro sphero manager')

    def run(self):
        """Start to run the websocket server in thread and get messages from the sensor."""
        self.running = True

        self.sphero_thread = threading.Thread(target=self.neurosphero.control_sphero)
        self.sphero_thread.daemon = True
        self.sphero_thread.start()

        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        print('running neuro sphero')

    def on_error(self, ws, error):
        print("ERROR: {0}".format(error))

    def on_close(self, ws):
        """Checks whether closed happened on purpose or not and handle it."""
        print("### websocket closed ###")
        if self.running is False:  # wanted disconnection
            print('Wanted disconnection')
            disconnect_neuro(sensor=self.sensor)  # close the connection to the neuro sensor and stop the recording.
            print('sent disconnect neuro')
        else:  # not wanted disconnection
            print('Unwanted disconnection')
            try:
                self.ws.close()  # Make sure websocket is really closed
            except Exception as e:
                print(e)

            self.login_neuro()  # login again and re-connect.
            self.ws = self.create_websocket_connection()
            self.run()

    def on_message(self, message):
        self.data = json.loads(message)#
        features = self.data[u'all']
        bafs = self.data[u'all'][1:122]
        self.neurosphero.buffer[self.neurosphero.sample_number] = bafs

        # check if data is valid
        qf = features[u'qf']
        if qf != 0:
            self.neurosphero.sphero_ball.set_color(255, 0, 0)
            print("data isn't valid!")

        # training mode
        if self.is_training:
            self.neurosphero.sphero_ball.set_color(255, 255, 255)  # white light

        # predict mode
        if (not self.is_training) and self.neurosphero.sample_number % 10 == 0:  # once every 10 samples make prediction
            sc = StandardScaler()
            self.neurosphero.buffer = sc.fit_transform(self.neurosphero.buffer)
            self.prediction = self.neurolearn.classifier.predict(self.neurosphero.buffer)
            self.prediction = (self.prediction > 0.5)
            indices = np.where(self.prediction)[0]
            self.prediction = self.prediction[indices]
            histogram = [0 for _ in range(4)]
            for p in self.prediction:
                histogram[np.argmax(p)] += 1
            if self.neurosphero.y_prediction == np.argmax(histogram):
                pass
            elif max(histogram) > 5:
                self.neurosphero.y_prediction = np.argmax(histogram)
            else:
                self.neurosphero.y_prediction = -1

        self.neurosphero.sample_number += 1
        if self.neurosphero.sample_number == 10:
            self.neurosphero.sample_number -= 10



    def login_neuro(self):
        """Login to neurosteer API"""
        login = NeuroLogin(email=self.email, password=self.password, sensor=self.sensor)
        login.get_token()
        return login

    def connect_sphero(self):
        """Connect to the sphero ball."""
        neurosphero = NeuroSphero(self.sphero_id)
        is_connected = neurosphero.connect()
        return neurosphero, is_connected

    def create_websocket_connection(self):
        """Create websocket connection to neurosteer API based on the token from login_neuro."""
        # C:\Users\owner\Anaconda2\Lib\site-packages\websocket\_logging.py
        # added null handler to avoid no handler error
        websocket.enableTrace(False)
        print("connecting to cloud...")

        ws = websocket.WebSocketApp(
            "wss://api.neurosteer.com/api/v1/features/" + self.sensor
            + "/real-time/?all=true&access_token=" + self.neuro.token,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        return ws

    def connect(self):
        """Loging to Neuro API using credentials and Sphero ball using the sphero id"""
        self.neuro = self.login_neuro()
        self.neurosphero, is_connected = self.connect_sphero()

        return self.create_websocket_connection()

    def disconnect(self):
        """Close the connection to neuro API and stop the recording."""
        self.running = False
        self.ws.close()

if __name__ == '__main__':
    neurosphero_manager = NeuroSpheroManager()
    neurosphero_manager.run()

