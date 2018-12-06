from flask import Flask
from flask import Response, request
import datetime as dt
import json

from main_sphero import NeuroSpheroManager
from main_sphero import EMAIL, PASSWORD, SENSOR, SPHERO_ID

# EMAIL = 'runtuchman@gmail.com'
# PASSWORD = '1234Ran'
# SENSOR = '00a3b4d8a9a7'
# SPHERO_ID = '68:86:e7:04:4d:10'


app = Flask(__name__, static_url_path='')


@app.route('/')
def index():
    return app.send_static_file(filename='index.html')


@app.route('/start-recording/', methods=['POST']) # describes what will happen when someone will get to thios path:
def start_recording():
    print('start recording')

    neurosphero_manager.run()

    description = json.loads(request.data)['description']
    description = '{} {:%d/%m/%y %H:%M}'.format(description,
                                                dt.datetime.today())
    neurosphero_manager.neuro.update_description(description=description)

    return Response(status=200)


@app.route('/stop-recording/')
def stop_recording():
    print('stop recording')

    try:
        neurosphero_manager.disconnect()
    except Exception as e:
        print(e)

    return Response(status=200)


@app.route('/reconnect-sphero/')
def reconnect_sphero():
    print('re connect sphero')
    is_connected = neurosphero_manager.connect_sphero()

    if is_connected:
        return Response(status=200)
    else:
        return Response(status=400)

if __name__ == '__main__':
    neurosphero_manager = NeuroSpheroManager()

    print("Init state: Stopping recording")
    try:
        neurosphero_manager.disconnect()
    except Exception as e:
        print(e)

    app.run(host='127.0.0.1', port=8000, debug=False)
