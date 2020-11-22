
from pynput.keyboard import Key, Listener
import csv
import time

csvfile = open("../data/keytrack.csv", "w", newline='')
writer = csv.writer(csvfile)
headers = ['events', 'time']
writer.writerow(headers)
    
def on_press(key):
    event = '{0} pressed'.format(key)
    print(event)
    row = event + ',' + str(time.time())
    # writer.writerow([event, str(time.time())])

def on_release(key):
    event = '{0} released'.format(key)
    print(event)
    if key == Key.esc:
        # Stop listener
        return False
    row = event + ',' + str(time.time())
    writer.writerow([event, str(time.time())])

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
