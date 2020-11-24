import logging

import EnvUtil


logging.getLogger().setLevel(logging.INFO)


class Agent:
    """
    Mother of all the agent models
    """

    def __init__(self, env, agent_type=None, test=False):
        self.env = env
        self.agent_type = agent_type
        self.agent = None

        if agent_type == "bc_agent":
            from model import bc_agent

            self.agent = bc_agent.BC_Agent()
            logging.info("using model: %s" % agent_type)
        elif agent_type == "traj_bc_agent":
            from model import traj_bc_agent

            self.agent = traj_bc_agent.TrajBCAgent(env, test=test)
            logging.info("using model: %s" % agent_type)

    def getaction(self):
        # AI agent works here
        # TODO: add the network and return the action

        testenv = self.env
        # sample code of getting data from the env
        print(testenv.getmap())
        print(testenv.getmapwidth())
        print(testenv.getmapheight())

        print(testenv.getchefpos())
        print(testenv.getorderlist())
        print(testenv.getchefholding())
        print(testenv.getobjposlist())
        print(testenv.getscore())
        """
        U: move up
        D: move down
        L: move left
        R: move right
        I: interact
        W: work action(chop, wash, etc.)
        """
        if self.agent is None:
            import random

            return ["U", "D", "L", "R", "I", "C"][random.choice(range(6))]
        elif self.agent_type == "bc_agent":
            states = EnvUtil.loss_less_encoding(testenv)
            return self.agent.action(states)
        elif self.agent_type == "traj_bc_agent":
            states = EnvUtil.loss_less_encoding(testenv)
            return self.agent.action(states)

    def train(self, filename):
        if self.agent_type == "traj_bc_agent":
            self.agent.train(
                filename, encoding_fn=EnvUtil.loss_less_encoding, epochs=500
            )

    def start(self):
        self.env.pyclient.start()
        while True:
            self.env.pyclient.update()

            chefid = 1
            action = self.getaction()
            # self.__sendaction(chefid, action)
            # time.sleep(3)
            obs, reward, done, env_info = self.env.step(action, chefid)
