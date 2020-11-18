import PyClient


# import Agent


class Env:
    """
    T: Table
    W: Work Station
    P: Plate Station
    D: Deliver Station
    0: Empty
    i: Ingredient i
    B: Food waste bin
    C: Cook Station
    N: NA
    """
    __map_width_data = {'1-1': 13, '1-2': 11}
    __map_height_data = {'1-1': 9, '1-2': 8}
    __map_data = {'1-1':
                      ['TTTTTNNNTTTTT',
                       'T00000000000T',
                       'B00000000000D',
                       'T00000000000D',
                       '100TT000TT00P',
                       'T000000000002',
                       'T00000000000T',
                       'TCC0000000CCT',
                       'NNN0000000NNN'],
                  '1-2':
                      ['TTWWWTTPDDT',
                       '00000000000',
                       '00000000000',
                       '00001230000',
                       '00000000000',
                       '00000000000',
                       '00000000000',
                       'TTTTBTCCCTT']}
    __x_base_data = {'1-1': 7.2, '1-2': 6.0}
    __z_base_data = {'1-1': -8.4, '1-2': 1.2}

    __cooradj = True
    __adjchefid = 1

    map_name = ''

    __chef_pos = [[0, 0, 0, 0], [0, 0, 0, 0]]

    pyclient = PyClient.PyClient()

    # agent = Agent.Agent()

    # Static data
    def __init__(self, n):
        self.map_name = n
        self.__map_width = self.__map_width_data[self.map_name]
        self.__map_height = self.__map_height_data[self.map_name]
        self.__map = self.__map_data[self.map_name]
        self.__step = 1.2
        self.__room = 0.4
        self.__x_base = self.__x_base_data[self.map_name]
        self.__z_base = self.__z_base_data[self.map_name]
        self.__x_border = self.__x_base - self.__step / 2
        self.__z_border = self.__z_base - self.__step / 2

    def getmap(self):
        return self.__map

    def getmapwidth(self):
        return self.__map_width

    def getmapheight(self):
        return self.__map_height

    # Dynamic data
    def getchefpos(self):
        self.__chef_pos = self.pyclient.getchefpos()
        x0 = int((self.__chef_pos[0][0] - self.__x_border) / self.__step)
        z0 = int((self.__chef_pos[0][2] - self.__z_border) / self.__step)
        x1 = int((self.__chef_pos[1][0] - self.__x_border) / self.__step)
        z1 = int((self.__chef_pos[1][2] - self.__z_border) / self.__step)
        '''
        Angle encoding:
        -45 to 45:  0  U
        45 to 135:  1  R
        135 to 225: 2  D
        225 to 315: 3  L
        '''
        a0 = int((self.__chef_pos[0][3] + 45) / 90)
        a0 = 0 if a0 == 4 else a0
        a1 = int((self.__chef_pos[1][3] + 45) / 90)
        a1 = 0 if a1 == 4 else a1
        return [[x0, z0, a0], [x1, z1, a1]]

    def getorderlist(self):
        return self.pyclient.getorderlist()

    def getchefholding(self):
        return self.pyclient.getchefholding()

    def getobjposlist(self):
        templist = self.pyclient.getobjposlist()
        outputlist = {}
        for itemtype in templist:
            outputlist[itemtype] = []
            for item in templist[itemtype]:
                x = int((item[0] - self.__x_border) / self.__step)
                z = int((item[1] - self.__z_border) / self.__step)
                if self.__cooradj:
                    [xc, zc] = self.getmapcellcenter(x, z)
                    if abs(xc - item[0]) >= 0.01 or abs(zc - item[1]) >= 0.01:
                        self.__chef_pos = self.pyclient.getchefpos()
                        x = int((self.__chef_pos[self.__adjchefid][0] - self.__x_border) / self.__step)
                        z = int((self.__chef_pos[self.__adjchefid][2] - self.__z_border) / self.__step)
                outputlist[itemtype].append([x, z])
        return outputlist

    def getpotprogress(self):
        return self.pyclient.getpotprogress()

    def getchopprogress(self):
        return self.pyclient.getchopprogress()

    def isfire(self):
        return self.pyclient.isfire()

    def getscore(self):
        return self.pyclient.getscore()

    def getmapcell(self, x, z):
        return self.__map[self.__map_height - 1 - z][x]

    def getmapcellcenter(self, x, z):
        return [(x + 0.5) * self.__step + self.__x_border, (z + 0.5) * self.__step + self.__z_border]
