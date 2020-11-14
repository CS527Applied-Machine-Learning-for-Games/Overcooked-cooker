import Env
import time
from Agent import Agent

class TestEnv(Env.Env):

    def __angleencoding(self, action):
        switcher = {
            'U': 0,
            'R': 1,
            'D': 2,
            'L': 3
        }
        return switcher.get(action, 'N')

    def __getcheffacing(self, chefid):
        x = self.getchefpos()[chefid][0]
        z = self.getchefpos()[chefid][1]
        a = self.getchefpos()[chefid][2]
        return [x if a % 2 == 0 else x + 2 - a, z if a % 2 != 0 else z + 1 - a]

    def __gettargetpos(self, chefid, a):
        x = self.getchefpos()[chefid][0]
        z = self.getchefpos()[chefid][1]
        return [x if a % 2 == 0 else x + 2 - a, z if a % 2 != 0 else z + 1 - a]

    def __sendaction(self, chefid, action):
        des_a = self.__angleencoding(action)
        [f_x, f_z] = self.__getcheffacing(chefid)
        if self.__angleencoding(action) == 'N':
            # interact or work
            if action == 'I':
                self.pyclient.pickdrop(chefid)
            elif action == 'C':
                self.pyclient.chop(chefid)
            else:
                print('ERROR: Undefine action code:', action)
        else:
            # move or rotate
            [des_x, des_z] = self.__gettargetpos(chefid, des_a)
            [c_x, c_z] = self.getmapcellcenter(des_x, des_z)
            if des_x < 0 or des_x >= self.getmapwidth() - 1 or des_z < 0 or des_z >= self.getmapheight() - 1:
                print('ERROR: Invalid destination.')
            elif self.getmapcell(des_x, des_z) == '0':
                self.pyclient.movechefto(chefid, c_x, c_z)
            else:
                # rotate
                self.pyclient.turn(chefid, des_a)
        
    
    def step(self, action, idx, sleep_time = 1):
        
        #send action and update
        self.pyclient.update()
        time.sleep(sleep_time)
        
        self.__sendaction(idx, action)
        
        time.sleep(sleep_time)
        self.pyclient.update()
        
        # return info
        obs = self.pyclient.getchefpos()
        reward = self.pyclient.getscore()
        done = self.pyclient.isfire()
        env_info = None
    
        return obs, reward, done, env_info
    
    
    def start(self):
        self.pyclient.start()
        self.agent = Agent(None)
        while True:
            self.pyclient.update()

            chefid = 1
            action = self.agent.getaction(self)
            
            # self.__sendaction(chefid, action)
            # time.sleep(3)
            obs, reward, done, env_info = self.step(action, chefid)