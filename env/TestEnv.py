import PyClient
import Agent


class TestEnv:
    map_name = ''

    __map_width = 11
    __map_height = 8
    __map = ['TTWWWTTPDDT',
             '00000000000',
             '00000000000',
             '00001230000',
             '00000000000',
             '00000000000',
             '00000000000',
             'TTTTBTCCCTT']
    '''
    T: Table
    W: Work Station
    P: Plate Station
    D: Deliver Station
    0: Empty
    i: Ingredient i
    B: Food waste bin
    C: Cook Station
    '''

    __step = 1.2
    __room = 0.4
    __x_base = 6.0
    __z_base = 1.2
    __x_border = __x_base-__step/2
    __z_border = __z_base-__step/2

    __chef_pos = [[0, 0, 0, 0], [0, 0, 0, 0]]

    pyclient = PyClient.PyClient()
    agent = Agent.Agent()

    # Static data
    def __init__(self, n):
        self.map_name = n

    def getmap(self):
        return self.__map

    def getmapwidth(self):
        return self.__map_width

    def getmapheight(self):
        return self.__map_height

    # Dynamic data
    def getchefpos(self):
        self.__chef_pos = self.pyclient.getchefpos()
        x0 = int((self.__chef_pos[0][0]-self.__x_border)/self.__step)
        z0 = int((self.__chef_pos[0][2]-self.__z_border)/self.__step)
        x1 = int((self.__chef_pos[1][0]-self.__x_border)/self.__step)
        z1 = int((self.__chef_pos[1][2]-self.__z_border)/self.__step)
        '''
        Angel encoding:
        -45 to 45:  0
        45 to 135:  1
        135 to 225: 2
        225 to 315: 3
        '''
        a0 = int((self.__chef_pos[0][3]+45)/90)
        a0 = 0 if a0 == 4 else a0
        a1 = int((self.__chef_pos[1][3]+45)/90)
        a1 = 0 if a1 == 4 else a1
        return [[x0, z0, a0], [x1, z1, a1]]

    def getorderlist(self):
        return self.pyclient.getorderlist()

    def getchefholding(self):
        return self.pyclient.getchefholding()

    def getobjposlist(self):
        return self.pyclient.getobjposlist()

    def getscore(self):
        return self.pyclient.getscore()

    def start(self):
        while True:
            self.pyclient.update()
            self.agent.getaction(self)
            self.pyclient.movechefto(0, 1, 2)
            self.pyclient.turn(0, 0)
            break
            # TODO: Categorize and execute the action
