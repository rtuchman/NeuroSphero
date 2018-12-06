import requests
import json
from dateutil.parser import parse
import pandas as pd
import numpy as np
import zipfile
import io
from NeuroLogin import *
import imageio
from PIL import Image

class NeuroProcess():

    def __init__(self):
        self.email = 'matanron3@gmail.com'
        self.password = 'Matan1234'
        self.sensor = '00a3b4810811'
        get_token = NeuroLogin(self.email, self.password, self.sensor)
        get_token.get_token()
        self.token = get_token.token
        #self.from_date = parse('2018-06-01')
        #self.to_date = parse('2019-06-01')

    def query_sessions(self, ns_connection,quary_string='', page=1, perPage=1000):
        payload = dict(access_token=self.token)
        url = ns_connection + '/api/v1/sensors/00a3b4810811/sessions/?page='+str(page)+'&searchTerm='+str(quary_string)+'&perPage='+str(perPage)
        resp = requests.get(url, params=payload)
        data = json.loads(resp.text)
        sessions = pd.DataFrame(data['sessions'])

        return sessions.sort_values('startDate')

    def save_data(self, ns_connection, sessionName, quary_string):
        url = ns_connection + r'/api/v1/sensors/' + self.sensor + r'/' + sessionName + r'/all?access_token='+self.token
        r = requests.get(url)
        buffer = io.BytesIO(r.content)
        z = zipfile.ZipFile(buffer)
        file = z.open('%s-%s.features.txt' % (self.sensor, sessionName))
        X = np.genfromtxt(file, delimiter=',')
        X = X[10:]
        X = X.T
        X = X[1:122]
        X = (X + 4) / 5
        X[X < 0] = 0
        X[X > 1] = 1
        for j in range(0, X.shape[1]-10, 10):
            img = Image.fromarray(X[:, j:j+10])
            img.save(r'C:\Users\owner\Desktop\NeuroSreer Project\dataset\train_data\{}\{}.{}.{}.tiff'.format(quary_string, quary_string, sessionName, j))
            print('printed {}.{}.{}'.format(quary_string, sessionName, j))



if __name__ == "__main__":
    query_list = ['MEMORY GAME'] #, 'CHILL MUSIC MEDITATE', 'WRITE WITH WEAK HAND']
    my = NeuroProcess()
    for q in query_list:
        sessions = my.query_sessions('https://api.neurosteer.com', q)
        for s in sessions.sessionName:
            my.save_data('https://api.neurosteer.com', s, q)

