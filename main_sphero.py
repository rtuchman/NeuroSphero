"""
   This module creates a NeuroLogin instance to get access to Neurosteer's api,
   then is creates a NeuroSphero instance that makes decisions and send commands
   to the sphero ball based on the measured data.


   After creating the instances, it opens a websocket with websocket.WebSocketApp.
   the websocket receives 3 args:
   url: websocket url.
   on_message: callable object which is called when received data. on_message has 2 arguments.
               The 1st argument is this class object.
               The 2nd argument is utf-8 string which we get from the server.
   on_error: callable object which is called when we get error. on_error has 2 arguments.
             The 1st argument is this class object.
             The 2nd argument is exception object.
   on_close: callable object which is called when closed the connection.
             this function has one argument. The argument is this class object.


   Args:
       email: user's email for api.neurosteer.com/signin.
       password: user's password for api.neurosteer.com/signin.
       sensor id: user's sensor.
       sphero id: user's sphero.
       features (optional): the biomarkers on which the user want perform analysis. default = (c1, h1).


   To run the program, connects sphero ball via bluetooth the the computer/rpi, wear the electrode and
   connect the sensor the the rpi, run in command line:
   python main_sphero.py <email> <password> <sensor> <sphero> <features(optional)>

"""
import threading
from functools import partial

import time

from NeuroSphero import *
from NeuroLogin import *
import sys
import websocket

from NeuroLogout import disconnect as disconnect_neuro

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

        self.running = False  # indication whether we want tp read data from the sensor or not
        self.ws = self.connect()
        print 'created neuro sphero manager'

    def run(self):
        """Start to run the websocket server in thread and get messages from the sensor."""
        print 'running neuro sphero'
        self.running = True

        ws_thread = threading.Thread(target=self.ws.run_forever)

        ws_thread.daemon = True
        ws_thread.start()

    def on_error(self, ws, error):
        print("ERROR: {0}".format(error))

    def on_close(self, ws):
        """Checks whether closed happened on purpose or not and handle it."""
        print "### websocket closed ###"
        if self.running is False:  # wanted disconnection
            print 'Wanted disconnection'
            disconnect_neuro(sensor=self.sensor)  # close the connection to the neuro sensor and stop the recording.
            print 'sent disconnect neuro'
        else:  # not wanted disconnection
            print 'Unwanted disconnection'
            try:
                self.ws.close()  # Make sure websocket is really closed
            except Exception as e:
                print e

            self.login_neuro()  # login again and re-connect.
            self.ws = self.create_websocket_connection()
            self.run()

    def on_message(self, ws, message):
        print 'message received'
        self.neurosphero.data = json.loads(message)
        features = self.neurosphero.data[u'features']
        # check if data is valid
        qf = features[u'qf']
        if qf != 0:
            self.neurosphero.sphero_ball.set_color(255, 0, 0)
            print "data isn't valid!"
        # training mode
        if self.neurosphero.sample_number <= self.neurosphero.calibration_samples:
            self.neurosphero.sphero_ball.set_color(255, 255, 255)  # white light  until buffer is full
        self.neurosphero.perform_calibration(features)
        if self.neurosphero.sample_number == self.neurosphero.calibration_samples + 1:
            print '\nCalibration is done'
        # controlling mode
        if self.neurosphero.sample_number > self.neurosphero.calibration_samples:
            print 'preform control sphero.'
            self.neurosphero.control_sphero(features)

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
        print "connecting to cloud..."

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
        self.neurosphero.buf = {feature: numpy.zeros([self.neurosphero.calibration_samples])
                                for feature in self.neurosphero.features}
        self.running = False
        self.ws.close()
