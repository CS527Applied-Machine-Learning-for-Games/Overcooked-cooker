import time
import TestEnv

testenv = TestEnv.TestEnv("1-2")
testenv.pyclient.start()
t = 1


def act(a, idx):
    if a in ["U", "D", "L", "R"]:
        testenv.sendaction(idx, a)
    elif a == "I":
        testenv.pyclient.pickdrop(idx)
    else:
        testenv.pyclient.chop(idx)


def step(action, idx):
    testenv.pyclient.update()
    time.sleep(t)
    act(action, idx)
    time.sleep(t)
    testenv.pyclient.update()

    obs = testenv.pyclient.getchefpos()
    reward = testenv.pyclient.getscore()
    done = testenv.pyclient.isfire()
    env_info = None

    return obs, reward, done, env_info


if __name__ == "__main__":
    import random

    steps = 0
    idx = 0
    while True:
        steps += 1
        print("=" * 10 + str(steps))
        c = ["U", "D", "L", "R", "I", "C"][random.choice(range(6))]
        step(c, idx)
