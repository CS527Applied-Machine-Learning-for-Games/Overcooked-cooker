import csv
import re

rows = []
with open("../data/data_0_gamedata.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        rows.append(row)

with open("../data/data_1_separated.csv", mode="w", newline="") as file:
    writer = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = [
        "player0_position_x",
        "player0_position_y",
        "player0_position_z",
        "player0_position_a",
        "player1_position_x",
        "player1_position_y",
        "player1_position_z",
        "player1_position_a",
        "order0",
        "order1",
    ]
    writer.writerow(header)
    for row in rows:
        player0_position = re.findall(r"Player 0: (.+?)Player 1:", row[0])
        player0_position_x = re.findall(r"x=(.+?),", player0_position[0])[0]
        player0_position_y = re.findall(r"y=(.+?),", player0_position[0])[0]
        player0_position_z = re.findall(r"z=(.+?),", player0_position[0])[0]
        player0_position_a = re.findall(r"a=(.+?) ", player0_position[0])[0]

        player1_position = re.findall(r"Player 1: (.+?)Order", row[0])
        player1_position_x = re.findall(r"x=(.+?),", player1_position[0])[0]
        player1_position_y = re.findall(r"y=(.+?),", player1_position[0])[0]
        player1_position_z = re.findall(r"z=(.+?),", player1_position[0])[0]
        player1_position_a = re.findall(r"a=(.+?) ", player1_position[0])[0]

        order0 = re.findall(r"Order:#1: (.+?) ", row[0])[0]
        order1 = re.findall(r"Order:#1: (.+?) ", row[0])[0]

        temp_row = [
            player0_position_x,
            player0_position_y,
            player0_position_z,
            player0_position_a,
            player1_position_x,
            player1_position_y,
            player1_position_z,
            player1_position_a,
            order0,
            order1,
        ]
        writer.writerow(temp_row)
