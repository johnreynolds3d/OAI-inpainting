import time

from main import main

s = time.time()
main(mode=2)
e = time.time()
print(e - s)
