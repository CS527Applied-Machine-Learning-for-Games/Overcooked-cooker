import logging

import EnvUtil
# from model import bc_agent

logging.getLogger().setLevel(logging.INFO)

class Agent:

    def __init__(self, agent_type = None):
        
        self.agent_type = agent_type
        self.agent = None
        
        if agent_type == "bc_agent":
            self.model = bc_agent.BC_Agent()
            logging.info("  using model: %s"%agent_type)
        

    def getaction(self, testenv):

        # AI agent works here
        # TODO: add the network and return the action

        # sample code of getting data from the env
        print(testenv.getmap())
        print(testenv.getmapwidth())
        print(testenv.getmapheight())

        print(testenv.getchefpos())
        print(testenv.getorderlist())
        print(testenv.getchefholding())
        print(testenv.getobjposlist())
        print(testenv.getscore())
        '''
        U: move up
        D: move down
        L: move left
        R: move right
        I: interact
        W: work action(chop, wash, etc.)
        '''
        if self.agent == None:
            import random
            return ['U','D','L','R','I','C'][random.choice(range(6))]
        elif self.agent_type == "bc_agent":
            states = EnvUtil.loss_less_encoding(testenv)
            return self.agent.action(states)
