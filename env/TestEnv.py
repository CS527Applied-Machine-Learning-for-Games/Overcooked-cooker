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
        Angle encoding:
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

    # Internal function
    def __angleencoding(self, action):
        switcher = {
            'U': 0,
            'R': 1,
            'D': 2,
            'L': 3
        }
        return switcher.get(action, 'N')

    def __getcheffacing(self, chefid):
        x = int((self.__chef_pos[chefid][0]-self.__x_border)/self.__step)
        z = int((self.__chef_pos[chefid][2]-self.__z_border)/self.__step)
        a = int((self.__chef_pos[chefid][3]+45)/90)
        a = 0 if a == 4 else a
        return [x if a % 2 == 0 else x + 2 - a, z if a % 2 != 0 else z + 1 - a]

    def __gettargetpos(self, chefid, a):
        x = int((self.__chef_pos[chefid][0]-self.__x_border)/self.__step)
        z = int((self.__chef_pos[chefid][2]-self.__z_border)/self.__step)
        return [x if a % 2 == 0 else x + 2 - a, z if a % 2 != 0 else z + 1 - a]

    def __getmapcell(self, x, z):
        return self.__map[self.__map_height - 1 - z][x]

    def __getmapcellcenter(self, x, z):
        return [(x + 0.5) * self.__step + self.__x_border, (z + 0.5) * self.__step + self.__z_border]

    def __sendaction(self, chefid, action):
        des_a = self.__angleencoding(action)
        [f_x, f_z] = self.__getcheffacing(chefid)
        if self.__angleencoding(action) == 'N':
            # interact or work
            if self.__getmapcell(f_x, f_z) == '0':
                print('ERROR: Try to interact or work on invalid cell.')
            elif action == 'I':
                self.pyclient.pickdrop(chefid)
            elif action == 'W':
                self.pyclient.work(chefid)
            else:
                print('ERROR: Undefine action code:', action)
        else:
            # move or rotate
            [des_x, des_z] = self.__gettargetpos(chefid, des_a)
            [c_x, c_z] = self.__getmapcellcenter(des_x, des_z)
            if self.__getmapcell(des_x, des_z) == '0':
                # move
                if des_x < 0 or des_x >= self.__map_width - 1 or des_z < 0 or des_z >= self.__map_height - 1:
                    print('ERROR: Invalid move.')
                else:
                    self.pyclient.movechefto(chefid, c_x, c_z)
            else:
                # rotate
                self.pyclient.turn(chefid, des_a)

    def start(self):
        while True:
            chefid = 0
            self.pyclient.update()
            action = self.agent.getaction(self)
            self.__sendaction(chefid, action)
