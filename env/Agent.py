class Agent:

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
        return 'W'
