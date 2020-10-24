import Env
import time


class Train(Env.Env):

    def start(self):
        self.pyclient.start()
        while True:
            self.pyclient.update()

            # TODO: record the data we need

            time.sleep(0.01)
