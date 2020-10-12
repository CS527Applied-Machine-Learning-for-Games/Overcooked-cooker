import pandas as pd
x_base = 4.8
z_base = 1.2
step = 1.2
room = 0.4

x_border = x_base-step/2
z_border = z_base-step/2

df = pd.read_csv('./data/data_1_separated.csv')

df['player0_position_x'] = (
    (df['player0_position_x']-x_border)/step).astype(int)
df['player0_position_z'] = (
    (df['player0_position_z']-z_border)/step).astype(int)
df['player0_position_a'] = ((df['player0_position_a']+45)/90).astype(int)
df.loc[df['player0_position_a'] == 4, ['player0_position_a']] = 0

df = df.drop(columns=['player0_position_y', 'player1_position_x',
                      'player1_position_y', 'player1_position_z', 'player1_position_a'])

#df['player1_position_x'] = ((df['player1_position_x']-x_border)/step).astype(int)
#df['player1_position_z'] = ((df['player1_position_z']-z_border)/step).astype(int)
#df['player1_position_a'] = ((df['player1_position_a']+45)/90).astype(int)
#df.loc[df['player1_position_a'] == 4, ['player1_position_a']] = 0

duplicate_row = [True]
for i in range(0, df.shape[0]-1):
    duplicate_row.append(not all((df.iloc[i] == df.iloc[i+1]).values))
df = df.loc[duplicate_row]

df['time'] = df.index

# print(df)
df.to_csv('./data/data_2_preprocessed.csv')
