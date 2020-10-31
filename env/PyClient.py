import socket
import time
import sys
import json
import random
from reprint import output


class PyClient:
    __chef_pos = [[0, 0, 0, 0], [0, 0, 0, 0]]
    __order_list = []
    __chefholding = []
    __objposlist = {}
    __score = 0

    def __del__(self):
        self.conn.close()
        self.s.close()

    def update(self):
        # send a request to C# server to update all the data in this class
        # chef_pos: float 0.00
        # order_list: string
        # chefholding: string
        self.conn.sendall(str.encode("request"))
        data = self.conn.recv(1024)
        if data:
            res = str(data)
            listdata = res.split(',')
            print(listdata)
            for j in range(4):
                self.__chef_pos[0][j] = (float(listdata[j + 1]))
            for j in range(4):
                self.__chef_pos[1][j] = (float(listdata[8 + j]))
            self.__order_list.clear()
            for s in listdata[15: 15 + int(listdata[14])]:
                self.__order_list.append(s)
            self.__chefholding.clear()
            self.__chefholding.append(listdata[6])
            self.__chefholding.append(listdata[13])
            self.__objposlist.clear()
           # print(int(listdata[15 + int(listdata[14])]))
            for s in range(int(listdata[15 + int(listdata[14])])):
                startindex = 15 + int(listdata[14]) + 1 + s * 4
                if listdata[startindex].lower().find("plate") != -1:
                    if self.__objposlist.get('Plate') is None:
                        self.__objposlist['Plate'] = [[float(listdata[startindex + 1]), float(listdata[startindex + 2]),
                                                       float(listdata[startindex + 3])]]

                    else:
                        pos = [float(listdata[startindex + 1]), float(listdata[startindex + 2]),
                               float(listdata[startindex + 3])]
                        self.__objposlist['Plate'].append(pos)
                if listdata[startindex].lower().find("pot") != -1:
                    if self.__objposlist.get('Pot') is None:
                        self.__objposlist['Pot'] = [[float(listdata[startindex + 1]), float(listdata[startindex + 2]),
                                                     float(listdata[startindex + 3])]]
                    else:
                        pos = [float(listdata[startindex + 1]), float(listdata[startindex + 2]),
                               float(listdata[startindex + 3])]
                        self.__objposlist['Pot'].append(pos)

        # self.__chef_pos = [[14.09, 0, 5.12, 0], [6.0, 0, 2.4, 180]]
        # self.__chef_pos = [[14.09, 0, 5.12, 0], [8.38, 0, 7, 180]]
        # self.__order_list = ['Sushi_Fish', 'Sushi_Fish']
        # self.__chefholding = ['Seaweed', 'Plate']

        # self.__objposlist = {'Plate': [[8.4, 13.2], [9.6, 13.2], [10.8, 13.2], [10.8, 13.2]], 'Suchi_rice': []}
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
    def pickdrop(self, chefid):
        # pick/drop action: press the pick/drop button once
        # return True when pick/drop successfully

        # TODO: send a request to C# server to perform the pick/drop action

        print('Chef ID:', chefid)
        print('Pickdrop')
        return True

    def work(self, chefid):
        # work action: press the work button once
        # return True when the work is finished

        # TODO: send a request to C# server to perform the work action

        print('Chef ID:', chefid)
        print('Work')
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

        msg = "action move " + str(chefid) + " " + str(current_x) + " " + str(current_z) + " " + str(x) + " " + str(z)
        self.conn.sendall(bytes(msg, encoding="utf-8"))
        return True

    def turn(self, chefid, direction):
        # turn action: turn the chef to face to the direction
        # turn action should be done when the chef is close to a Impassable location
        # only turn chef 0 for now
        current_x = self.__chef_pos[chefid][0]
        current_z = self.__chef_pos[chefid][2]
        print('Chef ID:', chefid)
        print('Current facing:', self.__chef_pos[chefid][3])
        print('Turn to face:', direction)

        # TODO: send a request to C# server to perform the turn action

        return True

    def start(self):
        HOST = ''
        PORT = 7777
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))
        self.s.listen(1)
        print('waiting...')
        self.conn, addr = self.s.accept()
        print('connected!')


if __name__ == "__main__":
    p = PyClient()
    p.start()

    while True:
        p.update()
        r1 = random.random();
        r2 = random.random();
        print("chef pos: ", p.getchefpos())
        print("chef holding: ", p.getchefholding())
        print("order list: ", p.getorderlist())
        print("obj poslist", p.getobjposlist())
        p.movechefto(1, 1 + r1 * 10, 5 + r2 * 10)
        time.sleep(1)
