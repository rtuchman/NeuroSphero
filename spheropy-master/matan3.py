from spheropy.Sphero import Sphero
from time import sleep
import random

print "one\n\n"


addrs = Sphero.find_spheros()
print "two\n\n"
s = Sphero(addrs.keys()[0], addrs.values()[0])
s.connect()
s.is_alive
for i in range(11):
    print"two\n\n"
    s.set_color(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    if (i%2 == 0 ):
        s.roll(30,0)
    else:
        s.roll(30,180)
    sleep(6)

