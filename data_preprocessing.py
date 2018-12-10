import requests
import json
from dateutil.parser import parse
import pandas as pd
import numpy as np
import zipfile
import io
from NeuroLogin import *
import imageio


class NeuroProcess():

    def __init__(self):
        self.email = 'matanron3@gmail.com'
        self.password = 'Matan1234'
        self.sensor = '00a3b4810811'
        get_token = NeuroLogin(self.email, self.password, self.sensor)
        get_token.get_token()
        self.token = get_token.token
        self.dataset = pd.DataFrame()
        #self.from_date = parse('2018-06-01')
        #self.to_date = parse('2019-06-01')

    def query_sessions(self, ns_connection,quary_string='', page=1, perPage=1000):
        payload = dict(access_token=self.token)
        url = ns_connection + '/api/v1/sensors/00a3b4810811/sessions/?page='+str(page)+'&searchTerm='+str(quary_string)+'&perPage='+str(perPage)
        resp = requests.get(url, params=payload)
        data = json.loads(resp.text)
        sessions = pd.DataFrame(data['sessions'])

        return sessions.sort_values('startDate')

    def save_data_as_image(self, ns_connection, sessionName, quary_string):
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
        X_normalized = X * 255
        for j in range(0, X_normalized.shape[1]-10, 10):
            buffer = X_normalized[:, j:j+10]
            imageio.imwrite(r'C:\Users\owner\Desktop\NeuroSreer Project\dataset\train_data\{}\{}.{}.{}.bmp'.format(quary_string, quary_string, sessionName, j), buffer)

    def save_data_as_csv(self, ns_connection, sessionName, y_index, y_size):

        url = ns_connection + r'/api/v1/sensors/' + self.sensor + r'/' + sessionName + r'/all?access_token='+self.token
        r = requests.get(url)
        buffer = io.BytesIO(r.content)
        z = zipfile.ZipFile(buffer)
        file = z.open('%s-%s.features.txt' % (self.sensor, sessionName))
        X = np.genfromtxt(file, delimiter=',')
        X = X[10:-10]
        X = X[:, 1:122]
        y = np.zeros((X.shape[0], y_size))
        y[:, y_index] = 1.0  # one hot label for the categorical data
        temp = np.concatenate((X, y), axis=1)
        temp_pd = pd.DataFrame(temp)
        self.dataset = pd.concat((self.dataset, temp_pd), axis=0)




if __name__ == "__main__":
    query_list = ['MEMORY GAME', 'CHILL MUSIC MEDITATE', 'WRITE WITH WEAK HAND', 'HAPPY MUSIC DANCING']
    my = NeuroProcess()
    for q in range(len(query_list)):
        sessions = my.query_sessions('https://api.neurosteer.com', query_list[q])
        for s in sessions.sessionName:
            my.save_data_as_csv('https://api.neurosteer.com', s, q, len(query_list))
            print('Saved: {} {}'.format(query_list[q], s))
    my.dataset.to_csv(r'neuro_data.csv')

