import Env
import time
import json
import copy
import os


class TrainEnv(Env.Env):

    def start(self):
        self.pyclient.start()
        time.sleep(2)
        prev_pos, cur_pos = [], []
        prev_hold, cur_hold = [], []
        prev_order, cur_order = [], []
        prev_chopping, cur_chopping = 0, 0

        with open("../data/test_1_1.json", "w") as f:
            while True:
                self.pyclient.update()

                cur_pos = self.getchefpos()
                cur_hold = self.getchefholding()
                cur_order = self.getorderlist()
                cur_chopping = sum(self.getchopprogress())

                if cur_pos != prev_pos or cur_hold != prev_hold or cur_order != prev_order or cur_chopping != prev_chopping:
                    new_state = {"pos": cur_pos, "hold": cur_hold, "order": cur_order, "objects": self.getobjposlist(),
                                 "score": self.getscore(), "pot": self.getpotprogress(), "chop": self.getchopprogress(),
                                 "fire": self.isfire()}

                    f.write(json.dumps(new_state) + '\n')
                    prev_pos = copy.deepcopy(cur_pos)
                    prev_hold = copy.deepcopy(cur_hold)
                    prev_order = copy.deepcopy(cur_order)
                    prev_chopping = cur_chopping

                time.sleep(0.1)


if __name__ == '__main__':
    # print(os.path.exists("../data/"))
    env = TrainEnv("1-1")
    env.start()
