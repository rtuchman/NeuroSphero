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

from NeuroSphero import *
from NeuroLogin import *
import sys
import websocket


def on_error(ws, error):
    print("ERROR: {0}".format(error))


def on_close(ws):
    url = ws.__dict__[u'url']
    print "### websocket closed ###"
    print "check that elctrode is on and that sensor is connected"
    print "check internet connection"
    print "trying to reconnect..."
    time.sleep(10)
    ws = websocket.WebSocketApp(
        url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close)
    ws.run_forever()


def on_message(ws, message):
    global neurosphero
    neurosphero.data = json.loads(message)
    features = neurosphero.data[u'features']
    # check if data is valid
    qf = features[u'qf']
    if qf != 0:
        neurosphero.sphero_ball.set_color(255, 0, 0)
        print "data isn't valid!"
    # training mode
    if neurosphero.sample_number <= neurosphero.calibration_samples:
        neurosphero.sphero_ball.set_color(255, 255, 255)  # white light  until buffer is full
    neurosphero.perform_calibration(features)
    # controlling mode
    if neurosphero.sample_number > neurosphero.calibration_samples:
        neurosphero.control_sphero(features)


if __name__ == "__main__":
    email = sys.argv[1]
    password = sys.argv[2]
    sensor = sys.argv[3]
    sphero = sys.argv[4]
    features = sys.argv[5:]

    login = NeuroLogin(email=sys.argv[1], password=sys.argv[2], sensor=sys.argv[3])
    login.get_token()
    token = login.token

    neurosphero = NeuroSphero(sphero)
    neurosphero.connect()

    # C:\Users\owner\Anaconda2\Lib\site-packages\websocket\_logging.py
    # added null handler to avoid no handler error
    websocket.enableTrace(False)
    print "connecting to cloud..."
    ws = websocket.WebSocketApp(
        "wss://api.neurosteer.com/api/v1/features/" + sensor
        + "/real-time/?all=true&access_token=" + token,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close)
    ws.run_forever()





