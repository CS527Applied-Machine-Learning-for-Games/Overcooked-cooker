class PyClient:
    __chef_pos = [[0, 0, 0, 0], [0, 0, 0, 0]]
    __order_list = []
    __chefholding = []
    __objposlist = {}
    __score = 0

    def update(self):

        # TODO: send a request to C# server to update all the data in this class

        self.__chef_pos = [[14.09, 0, 5.12, 0], [8.38, 0, 7, 180]]
        self.__order_list = ['Sushi_Fish', 'Sushi_Fish']
        self.__chefholding = ['Seaweed', 'Plate']
        self.__objposlist = {'Plate': [[8.4, 13.2], [9.6, 13.2], [10.8, 13.2], [10.8, 13.2]],
                             'Suchi_rice': []}
        self.__score = 30

    # get data from the game
    def getchefpos(self):
        return self.__chef_pos

    def getorderlist(self):
        return self.__order_list

    def getchefholding(self):
        return self.__chefholding

    def getobjposlist(self):
        return self.__objposlist

    def getscore(self):
        return self.__score

    # take action
    def pickdrop(self):
        # pick/drop action: press the pick/drop button once
        # return True when pick/drop successfully

        # TODO: send a request to C# server to perform the pick/drop action

        return True

    def work(self):
        # work action: press the work button once
        # return True when the work is finished

        # TODO: send a request to C# server to perform the work action

        return True

    def movechefto(self, chefid, x, z):
        # move action: move the chef to (x,z)
        # (x,z) should be within one step to the pos in __chef_pos
        # only move chef 0 for now
        current_x = self.__chef_pos[chefid][0]
        current_z = self.__chef_pos[chefid][2]
        print('Chef ID:', chefid)
        print('Current pos:', current_x, current_z)
        print('Target pos:', x, z)

        # TODO: send a request to C# server to perform the move action

        return True
