import TestEnv
import Agent

testenv = TestEnv.TestEnv('1-2')

agent = Agent.Agent(testenv)
agent.start()


# TODO
# -1. saved data process (remove intermedia states, discrete)
# -2. check action finished
# -3. key/action recording/inferring
# 4. new BC agent (include encoding)
# ?5. new AC agent
# x. turn
