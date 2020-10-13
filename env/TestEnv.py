import PyClient


class TestEnv:
    map_name = ''
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
    __map_width = 11
    __map_height = 8

    __chef_pos = [[0, 0, 0, 0], [0, 0, 0, 0]]

    pyclient = PyClient.PyClient()

    __score = 0

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
        __chef_pos = self.pyclient.getchefpos()
        return self.__chef_pos

    def getscore(self):
        __score = self.pyclient.getscore()
        return __score
