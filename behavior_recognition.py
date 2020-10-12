import pandas as pd


def rotate(i):
    switcher = {
        0: 'Up',
        1: 'Right',
        2: 'Down',
        3: 'Left'
    }
    return switcher.get(i, "Invalid")


df = pd.read_csv('./data/data_2_preprocessed.csv')

print(df)

move_action = []
cut_action = []
pickdrop_action = []

diff_x = 0
diff_z = 0

for i in range(0, df.shape[0]-1):
    diff_x = df.iloc[i+1]['player0_position_x'] - \
        df.iloc[i]['player0_position_x']
    diff_z = df.iloc[i+1]['player0_position_z'] - \
        df.iloc[i]['player0_position_z']

    if (diff_x and diff_z):
        move_action.append(rotate(-1))
    elif (diff_x == 0 and diff_z == 0):
        move_action.append(rotate(df.iloc[i+1]['player0_position_a']))
    elif (diff_x):
        move_action.append(rotate(2-diff_x))
    elif (diff_z):
        move_action.append(rotate(1-diff_z))

    cut_action.append(abs(df.iloc[i]['time']-df.iloc[i+1]['time']) > 100)
    pickdrop_action.append(False)

move_action.append(rotate(df.iloc[df.shape[0]-1]['player0_position_a']))
cut_action.append(False)
pickdrop_action.append(False)
df['move_action'] = move_action
df['cut_action'] = cut_action
df['pickdrop_action'] = pickdrop_action

# print(df)
df.to_csv('./data/data_3_w_action.csv')
