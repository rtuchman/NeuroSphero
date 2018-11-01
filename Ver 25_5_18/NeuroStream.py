import requests
from urllib2 import urlparse
import json
import websocket

class NeuroStream:
    """A simple example class"""
    token = ''
    btname = ''
    on_message = []

    def __init__(self):
        return

    @staticmethod
    def connect(email, password, sensor):
        self = NeuroStream()
        r = requests.post("https://api.neurosteer.com/signin",
                          data={'email': email, 'password': password})
        data = json.loads(r.text)
        response_url = data[u'url']
        parsed = urlparse.urlparse(response_url)

        self.token = urlparse.parse_qs(parsed.query)['access_token'][0]
        self.btname = sensor
        return self

    def on_err(ws, error):
        print("stream err " + error)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return

    def stream(self, on_message, on_error =on_err):
        self.on_message = on_message
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(
            "wss://api.neurosteer.com/api/v1/features/" + self.btname + "/real-time/?all=true&access_token=" + self.token,
            on_message=on_message,
            on_error=on_error,
            on_close=self.on_close)

        ws.run_forever()



    def on_close(self, ws):
        print("### closed ###")
        print("trying to reconnect...")
        ws = websocket.WebSocketApp(
            "wss://api.neurosteer.com/api/v1/features/" + self.btname + "/real-time/?all=true&access_token=" + self.token,
            on_message=self.on_message,
            on_error=self.on_err,
            on_close=self.on_close)

        ws.run_forever()


def on_message_example(ws, message):
    # print message
    message = json.loads(message)
    biomarkers = message[u'features']
    baf = message[u'all'][:121]
    c1 = biomarkers[u'c1']
    baf1 = baf[0]
    print("c1 biomarker: "+ '%.2f' % c1 +", first BAF: "+ '%.2f' % baf1)

if __name__ == "__main__":
    with NeuroStream.connect(email='mymail', password='mypassword', sensor='sensorname') as ns:
        ns.stream(on_message_example)