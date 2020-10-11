class PyClient:
    __chef_pos = [[0, 0, 0, 0], [0, 0, 0, 0]]
    __order_list = []
    
    def update(self):
        self.__chef_pos = [[14.09, 0, 5.12, 0], [8.38, 0, 7, 180]]
        self.__order_list = ['Sushi_Fish', 'Sushi_Fish']

    def getchefpos(self):
        return self.__chef_pos
    def getorderlist(self):
        return self.__order_list