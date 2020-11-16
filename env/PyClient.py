import socket
import time
import sys
import json
import random
import csv
import copy


class PyClient:
    __chef_pos = [[0., 0., 0., 0.], [0., 0., 0., 0.]]
    __order_list = []
    __chefholding = []
    __objposlist = {}
    __score = 0
    __isfire = 'False'
    __potprogress = []
    __chopprogress = []

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
        # print("updating information")
        if data:
            res = str(data)
            listdata = res.split(',')
            # print(listdata)
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
            global startindex
            global itemstartindex
            global nextindex
            global pos
            foodcount = {}
            for s in range(int(listdata[15 + int(listdata[14])])):
                startindex = 15 + int(listdata[14]) + 1 + s * 4
                pos = [float(listdata[startindex + 2]), float(listdata[startindex + 3])]
                if listdata[startindex].lower().find("plate") != -1:
                    if self.__objposlist.get('Plate') is None:
                        self.__objposlist['Plate'] = [pos]
                    else:
                        self.__objposlist['Plate'].append(pos)
                if listdata[startindex].lower().find("pot") != -1:
                    if self.__objposlist.get('Pot') is None:
                        self.__objposlist['Pot'] = [pos]
                    else:
                        self.__objposlist['Pot'].append(pos)
                for s in listdata[startindex + 1].split('+'):
                    if s != "None":
                        if foodcount.__contains__(s):
                            foodcount[s] = foodcount[s] + 1
                        else:
                            foodcount.setdefault(s, 1)
                        if self.__objposlist.get(s + "_" + str(foodcount[s])) is None:
                            self.__objposlist[s + "_" + str(foodcount[s])] = [pos]
                        else:
                            self.__objposlist[s + "_" + str(foodcount[s])].append(pos)
#                         if self.__objposlist.get('Food') is None:
#                             self.__objposlist['Food'] = [pos]
#                         else:
#                             self.__objposlist['Food'].append(pos)

            startindex = startindex + 4
            if int(listdata[startindex]) != 0:
                for s in range(int(listdata[startindex])):
                    itemstartindex = startindex + 1 + s * 3
                    pos = [float(listdata[itemstartindex + 1]), float(listdata[itemstartindex + 2])]
#                     if self.__objposlist.get('Food') is None:
#                         self.__objposlist['Food'] = [pos]
#                     else:
#                         self.__objposlist['Food'].append(pos)

                    if foodcount.__contains__(listdata[itemstartindex]):
                        foodcount[listdata[itemstartindex]] = foodcount[listdata[itemstartindex]] + 1
                    else:
                        foodcount.setdefault(listdata[itemstartindex], 1)
                    if self.__objposlist.get(listdata[itemstartindex] + "_" + str(foodcount[listdata[itemstartindex]])) is None:
                        self.__objposlist[listdata[itemstartindex] + "_" + str(foodcount[listdata[itemstartindex]])] = [pos]
                    else:
                        self.__objposlist[listdata[itemstartindex] + "_" + str(foodcount[listdata[itemstartindex]])].append(pos)
            pos = []
            if int(listdata[startindex]) == 0:
                nextindex = startindex + 1
            else:
                nextindex = itemstartindex + 3
            self.__potprogress.clear()
            for s in range(int(listdata[nextindex])):
                self.__potprogress.append(float(listdata[nextindex + s + 1]))
            nextindex = nextindex + 1 + int(listdata[nextindex])
            self.__isfire = listdata[nextindex]
            nextindex = nextindex + 1
            self.__score = int(listdata[nextindex])
            nextindex = nextindex + 1
            self.__chopprogress.clear()
            for s in range(int(listdata[nextindex])):
                self.__chopprogress.append(float(listdata[nextindex + 1 + s]))
            return res

        # self.__chef_pos = [[14.09, 0, 5.12, 0], [6.0, 0, 2.4, 180]]
        # self.__chef_pos = [[14.09, 0, 5.12, 0], [8.38, 0, 7, 180]]
        # self.__order_list = ['Sushi_Fish', 'Sushi_Fish']
        # self.__chefholding = ['Seaweed', 'Plate']
        # self.__objposlist = {'Plate': [[8.4, 13.2], [9.6, 13.2], [10.8, 13.2], [10.8, 13.2]],
        #                      'Suchi_rice': []}
        # self.__score = 30

    # get data from the game
    def getchefpos(self):
        return self.__chef_pos

    def getorderlist(self):
        return self.__order_list

    def getchefholding(self):
        return self.__chefholding

    def getobjposlist(self):
        return self.__objposlist

    def getpotprogress(self):
        return self.__potprogress

    def getchopprogress(self):
        return self.__chopprogress

    def isfire(self):
        return self.__isfire

    def getscore(self):
        return self.__score

    # take action
    def pickdrop(self, chefid):
        # pick/drop action: press the pick/drop button once
        # return True when pick/drop successfully

        # TODO: send a request to C# server to perform the pick/drop action

        print('Chef ID:', chefid)
        print('Pickdrop')
        msg = "action pickdrop " + str(chefid)
        self.conn.sendall(bytes(msg, encoding="utf-8"))
        return True

    def chop(self, chefid):
        # work action: press the work button once
        # return True when the work is finished

        # TODO: send a request to C# server to perform the work action

        print('Chef ID:', chefid)
        print('Work')

        msg = "action chop " + str(chefid)
        self.conn.sendall(bytes(msg, encoding="utf-8"))
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

        msg = "action move " + str(chefid) + " " + str(current_x) + \
            " " + str(current_z) + " " + str(x) + " " + str(z)
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
    time.sleep(2)
    prev_pos, cur_pos = [], []
    prev_hold, cur_hold = [], []
    prev_order, cur_order = [], []

    with open("../data/rawdata.txt", "w") as csvfile:
        # writer = csv.writer(csvfile)
        # headers = ['res']
        # writer.writerow(headers)

        while True:
            csvfile.write(p.update() + '\n')
            '''
            r1 = random.random()
            r2 = random.random()
            print("chef pos: ", p.getchefpos())
            print("chef holding: ", p.getchefholding())
            print("order list: ", p.getorderlist())
            print("obj poslist", p.getobjposlist())
            print("score: ", p.getscore())
            # p.movechefto(1, 1 + r1 * 10, 5 + r2 * 10)
            # p.pickdrop(1)
            # p.chop(1)
            cur_pos = p.getchefpos()
            cur_hold = p.getchefholding()
            cur_order = p.getorderlist()
            print("cur pos: ", cur_pos, 'prev pos: ', prev_pos)
            print("cur holding: ", cur_hold, 'prev holding:', prev_hold)
            print("order list: ", cur_order, 'prev list: ', prev_order)
            print("pot states: ", p.getpotprogress())
            print("chop states: ", p.getchopprogress())
            print("fire", p.isfire())
            if cur_pos != prev_pos or cur_hold != prev_hold or cur_order != prev_order:
                print('record ')
                writer.writerow(cur_pos+cur_hold+cur_order)
                prev_pos = copy.deepcopy(cur_pos)
                prev_hold = copy.deepcopy(cur_hold)
                prev_order = copy.deepcopy(cur_order)
            '''
            time.sleep(0.05)
