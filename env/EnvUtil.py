import TestEnv
import numpy as np
from collections import OrderedDict
"""
Functions here used to encode states of the environment to feed ai agents.
"""

base_map_features = ['T', 'W', 'P', 'D', 'i', 'B', 'C'] # Table, Work Station, Plate Station, Deliver Station, Ingredient i, Food waste, Cook Station
variable_map_features = ['player_location', 
                         'plate', 'pot',
                         'rice', 'fish', 'seaweed',
                         'rice_in_pot', 'fish_chopped', 
                         'plates_with_seaweed', 'plates_with_fish', 'plates_with_rice',
                         'plates_with_rice_seaweed', 'plates_with_fish_seaweed', 'plates_with_fish_rice',
                         'plates_with_fish_rice_seaweed']

def loss_less_encoding(env):
    env_map = np.asarray([[c for c in i] for i in env.getmap()])
    encoding = []
    
    # base_map_features
    # Encode position of Table, Work Station, Plate Station, Deliver Station, Ingredient i, Food waste, Cook Station 
    for label in base_map_features:
        encoding.append((env_map == label).astype(int))
        
    height = env.getmapheight()
    # variable_map_features    
    variable_map_layers = OrderedDict()
    for v in variable_map_features:
        variable_map_layers[v] = np.zeros(np.shape(env_map))
    # player location
    player_idx = 1
    player_pos = env.getchefpos()[player_idx]
    variable_map_layers['player_location'][height - 1 - player_pos[1]][player_pos[0]] = 1
    # obj
    objs = env.getobjposlist()
    for obj in objs:
        
        if obj == 'Plate':
            for pos in objs[obj]:
                variable_map_layers['plate'][height - 1 - pos[1]][pos[0]] = 1
                
        elif obj == 'Pot':
            for pos in objs[obj]:
                variable_map_layers['pot'][height - 1 - pos[1]][pos[0]] = 1
                
        elif obj.startswith('Seaweed'):
            for pos in objs[obj]:
                if pos in objs['Plate']:
                    variable_map_layers['plates_with_seaweed'][height - 1 - pos[1]][pos[0]] = 1
                else:
                    variable_map_layers['seaweed'][height - 1 - pos[1]][pos[0]] = 1
                    
        elif obj.startswith('SushiFish'):
            for pos in objs[obj]:
                if pos in objs['Plate']:
                    variable_map_layers['plates_with_fish'][height - 1 - pos[1]][pos[0]] = 1
                else:
                    variable_map_layers['fish'][height - 1 - pos[1]][pos[0]] = 1
                
        elif obj.startswith('SushiRice'):
            for pos in objs[obj]:
                if pos in objs['Pot']:
                    # this could be an urgent layer, how about making it all one ?? 
                    variable_map_layers['rice_in_pot'][height - 1 - pos[1]][pos[0]] = 1
                elif pos in objs['Plate']:
                    variable_map_layers['plates_with_rice'][height - 1 - pos[1]][pos[0]] = 1
                else:
                    variable_map_layers['rice'][height - 1 - pos[1]][pos[0]] = 1
                    
        elif obj.startswith('ChoppedSushiFish'):
            for pos in objs[obj]:
                variable_map_layers['fish_chopped'][height - 1 - pos[1]][pos[0]] = 1
    
    
    seaweed_rice_fish = variable_map_layers['plates_with_seaweed'] * variable_map_layers['plates_with_rice'] * variable_map_layers['plates_with_fish']
    if np.any(seaweed_rice_fish):
        variable_map_layers['plates_with_fish_rice_seaweed'] = seaweed_rice_fish
        variable_map_layers['plates_with_fish'] -= seaweed_rice_fish
        variable_map_layers['plates_with_seaweed'] -= seaweed_rice_fish
        variable_map_layers['plates_with_rice'] -= seaweed_rice_fish
        
    fish_rice = variable_map_layers['plates_with_rice'] * variable_map_layers['plates_with_fish']
    if np.any(fish_rice):
        variable_map_layers['plates_with_fish_rice'] = fish_rice
        variable_map_layers['plates_with_rice'] -= fish_rice
        variable_map_layers['plates_with_fish'] -= fish_rice
    
    fish_seaweed = variable_map_layers['plates_with_fish'] * variable_map_layers['plates_with_seaweed']
    if np.any(fish_seaweed):
        variable_map_layers['plates_with_fish_seaweed'] = fish_seaweed
        variable_map_layers['plates_with_fish'] -= fish_seaweed
        variable_map_layers['plates_with_seaweed'] -= fish_seaweed
    
    rice_seaweed = variable_map_layers['plates_with_rice'] * variable_map_layers['plates_with_seaweed']
    if np.any(rice_seaweed):
        variable_map_layers['plates_with_rice_seaweed'] = rice_seaweed
        variable_map_layers['plates_with_rice'] -= rice_seaweed
        variable_map_layers['plates_with_seaweed'] -= rice_seaweed
    
    # print(variable_map_layers)
    
    for layer_id in variable_map_layers:
        encoding.append(variable_map_layers[layer_id])
    encoding_stack = np.stack(encoding)
    encoding_stack = np.transpose(encoding_stack, (1, 2, 0))
    return encoding_stack.astype(int)
    
    
def get_loss_less_encoding_shape(env = None):
    if env is None:
        return (len(base_map_features) + len(variable_map_features), )
    else:
        return (len(base_map_features) + len(variable_map_features), env.getmapheight(), env.getmapwidth())
    
    
def __test_loss_less_encoding(env):
    env_map = np.asarray([[c for c in i] for i in env.getmap()])
    encoding = loss_less_encoding(env)
    for i in range(len(base_map_features)):
        expected = (env_map == base_map_features[i]).astype(int)
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


def __test_featurized_encoding(env):
    encoding = featurized_encoding(env)
    assert(get_featurized_encoding_shape() == np.shape(encoding))
    print('shape of loss less encoding:', np.shape(encoding))

    
if __name__ == "__main__":
    env = TestEnv.TestEnv('1-2')
    __test_loss_less_encoding(env)
    __test_featurized_encoding(env)