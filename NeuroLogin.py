"""Module for setting up connection with Neurosteer API"""

import requests
from urlparse import (urlparse, parse_qs)
import json
import time
from pprint import pprint as pp


class NeuroLogin:
    """A class that retrieves access token from api.neurosteer.com

       Args:
            email: user's email for api.neurosteer.com/signin
            password: user's password for api.neurosteer.com/signin
            sensor id: user's sensor
    """
    def __init__(self, email, password, sensor):
        self.email = email
        self.password = password
        self._sensor = sensor
        return

    def get_token(self):
        try:
            r = requests.post("https://api.neurosteer.com/signin",
                              data={'email': self.email, 'password':  self.password})
            data = json.loads(r.text)
            response_url = data[u'url']
            parsed = urlparse(response_url)
            access_token = parse_qs(parsed.query)['access_token'][0]
            self.token = access_token
            print "token successfully retrieved"
        except KeyError:
            print "Invalid email or password"
            print "Please re-enter email:"
            self.email = str(raw_input())
            print "Please re-enter password:"
            self.password = str(raw_input())
            self.get_token()
        return




