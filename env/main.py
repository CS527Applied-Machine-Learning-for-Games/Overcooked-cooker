import TestEnv
import Agent


def test_random_run():
    testenv = TestEnv.TestEnv('1-2')

    agent = Agent.Agent(testenv)
    agent.start()

def test_traj_bc_train():
    testenv = TestEnv.TestEnv('1-1')

    agent = Agent.Agent(testenv, agent_type="traj_bc_agent")
    agent.train()

def test_traj_bc():
    testenv = TestEnv.TestEnv('1-1')

    agent = Agent.Agent(testenv, agent_type="traj_bc_agent", test=True)
    agent.start()


# TODO
# -1. saved data process (remove intermedia states, discrete)
# -2. check action finished
# -3. key/action recording/inferring
# 4. new BC agent (include encoding)
# ?5. new AC agent
# -x. turn


if __name__ == '__main__':
    # test_random_run()
    # test_traj_bc_train()
    test_traj_bc()
