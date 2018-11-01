import spheropy.Sphero
#import time
import websocket
import threading
import requests
import json
import urlparse
from websocket import create_connection

# sensor_id = '00a3b4d8a99e'
# r = requests.post("https://api.neurosteer.com/signin", data={'email': 'runtuchman@gmail.com', 'password': '841956rt!'})
# print(r.status_code, r.reason)
# print(r.text)
# data = json.loads(r.text)
# print(data)
# response_url = data[u'url']
# parsed = urlparse.urlparse(response_url)
# token = urlparse.parse_qs(parsed.query)['access_token'][0]
#
# speed = 0


#if sensor is not connected don't do 
def on_message(ws, message):
    global k
    global thresh
    nbuf = len(buf);
    #print message
    message = json.loads(message)
    features =  message[u'features']
    c1 = features[u'c1']
    h1 = features[u'h1']
    buf[k] = c1
    k = (k+1)%nbuf
    c1_ma = sum(buf)/float(nbuf)
    c1_ma = (c1_ma+1)*50
    print c1_ma
    speed = 0
    #print h1
    if c1_ma < thresh[0]:
    # speed = 50 * (1 - c1)
        sphero.set_color(0, 255, 0)
    elif c1_ma > thresh[1]:
        sphero.set_color(255, 0, 0)
    else:
        sphero.set_color(0, 0, 255)
    # sphero.roll(speed, 0)


def on_error(ws, error):
    print error

def on_close(ws):
    print "### closed ###"

sphero = None;
buf_size = 10
buf = [0]*buf_size
thresh = [73, 80]
k = 0


if __name__ == "__main__":

        #OBR: "68:86:e7:01:fb:b2"
        #YPR: "68:86:e7:04:4d:10"

 
    with Sphero("NAME", "68:86:e7:01:fb:b2") as s:
        #global sphero
        sphero=s
        response = sphero.ping()
        print(response)

        #sphero.set_color(0, 0, 255)
        #sphero.roll(speed, 0)
        #time.sleep(5)

        websocket.enableTrace(False)
        ws = websocket.WebSocketApp("wss://api.neurosteer.com/api/v1/features/"+sensor_id+"/real-time/?access_token="+token,
                                    on_message = on_message,
                                    on_error = on_error,
                                    on_close = on_close)

        ws.run_forever()






