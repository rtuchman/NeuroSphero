import NeuroSphero
import sys
from spheropy.Sphero import Sphero
from time import sleep
if __name__ == "__main__":
    for arg in sys.argv[1:]:
            print(arg)
    email = sys.argv[1] # set as matanron3@gmail.com
    password = sys.argv[2] # set as Matan1234
    sensor = sys.argv[3] # set as 00a3b4d8a99e
    # sphero = sys.argv[4]
    # features = sys.argv[5:]


    # addrs = Sphero.find_spheros() # since we only have one Sphero right now I connect directly to it and not look for other balls so that the run attempts will take less time
    sphero_address = '68:86:E7:04:4D:10' # addrs.values()[0]
    sphero_name = u'Sphero-YPR'

    print "found"
    s = Sphero(sphero_name, sphero_address)
    s.connect()
    print "connected"

    s.set_color(255, 255, 255)
    sleep(2)
    s.set_color(255, 0, 0)
    sleep(2)
    s.set_color(0, 0, 255)
    sleep(2)
    s.set_color(0, 0, 255)
    sleep(2)
    #print "white light"

    # with NeuroSphero.NeuroSphero.connect(email=email, password=password, sensor=sensor, sphero_address=sphero_address) as ns:
    #     ns.stream()