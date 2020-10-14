import TestEnv
import numpy as np

"""
Functions here used to encode states of the environment to feed ai agents.
"""

pos_label = ['T', 'W', 'P', 'D', 'i', 'B', 'C'] # Table, Work Station, Plate Station, Deliver Station, Ingredient i, Food waste, Cook Station


def loss_less_encoding(env):
    env_map = np.asarray([[c for c in i] for i in env.getmap()])
    encoding = []
    # Encode position of Table, Work Station, Plate Station, Deliver Station, Ingredient i, Food waste, Cook Station 
    for label in pos_label:
        encoding.append((env_map == label).astype(int))
    return np.stack(encoding)
    
    
def get_loss_less_encoding_shape(env = None):
    if env is None:
        return (len(pos_label), )
    else:
        return (len(pos_label), env.getmapheight(), env.getmapwidth())
    
    
def test_loss_less_encoding(env):
    env_map = np.asarray([[c for c in i] for i in env.getmap()])
    encoding = loss_less_encoding(env)
    for i in range(len(pos_label)):
        expected = (env_map == pos_label[i]).astype(int)
        assert((expected == encoding[i]).all() == True)
    assert(np.shape(encoding) == get_loss_less_encoding_shape(env))
    print('shape of loss less encoding:', np.shape(encoding))
    

def featurized_encoding(env):
    
    encoding = []
    
    # orientation
    orientation = [0]*4
    # TODO: get orientation
    # orientation = orientation[env.getorientation()] = 1
    encoding.extend(orientation)
    
    #holding object
    holdind_object = [0]*4
    env.getchefholding()
    # TODO: onehot holding / embedded holding list
    encoding.extend(holdind_object)
    
    return np.asarray(encoding)
    
    
def get_featurized_encoding_shape():
    return (4 + 4, )


def test_featurized_encoding(env):
    encoding = featurized_encoding(env)
    assert(get_featurized_encoding_shape() == np.shape(encoding))
    print('shape of loss less encoding:', np.shape(encoding))

    
if __name__ == "__main__":
    env = TestEnv.TestEnv('1-2')
    test_loss_less_encoding(env)
    test_featurized_encoding(env)