import json
import copy

dir2action = {
    (1, 0): "R",
    (-1, 0): "L",
    (0, 1): "U",
    (0, -1): "D",
    0: "U",
    1: "R",
    2: "D",
    3: "L"
}

action2face = {
    "U": 0,
    "R": 1,
    "D": 2,
    "L": 3
}

face2lastmove = {
    0: (1, 0),
    1: (-1, 0),
    2: (0, 1),
    3: (0, -1)
}

invalid_cells = {"1-1": {(3, 3), (4, 3), (8, 3), (9, 3)},
                 "1-2": {(4, 4), (5, 4), (6, 4)}}


def cal_dis(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def infer_acton(infile, outfile, level):
    # only for chef 1
    with open(infile) as fi:
        in_lines = list(map(json.loads, fi.readlines()))
        out_lines = []
        for i in range(len(in_lines) - 1):
            cur_state = in_lines[i]
            next_state = in_lines[i + 1]
            miss_state = None
            if cur_state["pos"] != next_state["pos"]:
                dx = next_state["pos"][1][0] - cur_state["pos"][1][0]
                dz = next_state["pos"][1][1] - cur_state["pos"][1][1]
                a = next_state["pos"][1][2]
                if abs(dx) + abs(dz) > 1 and cur_state["hold"] == next_state["hold"]:
                    # print("step {} missed".format(i + 1))
                    if abs(dx) + abs(dz) == 2:
                        # miss one move
                        miss_state = copy.deepcopy(cur_state)
                        if (abs(dx) == 2 or abs(dz) == 2) and (
                        cur_state["pos"][1][0] + dx // 2, cur_state["pos"][1][1] + dz // 2) not in invalid_cells[level]:
                            miss_state["pos"][1] = [miss_state["pos"][1][0] + dx // 2,
                                                    miss_state["pos"][1][1] + dz // 2,
                                                    action2face[dir2action[(dx // 2, dz // 2)]]]
                            miss_action = action = dir2action[(dx // 2, dz // 2)]
                        else:
                            inferred_lastmove = face2lastmove[next_state["pos"][1][2]]
                            inferred_pos = (next_state["pos"][1][0] - inferred_lastmove[0],
                                            next_state["pos"][1][1] - inferred_lastmove[1])
                            if cal_dis(inferred_pos, cur_state["pos"][1]) != 1:
                                # if infered next move not next to cur pos
                                inferred_lastmove = (dx, 0)
                                inferred_pos = (next_state["pos"][1][0] - inferred_lastmove[0],
                                                next_state["pos"][1][1] - inferred_lastmove[1])

                            if inferred_pos not in invalid_cells[level]:
                                miss_state["pos"][1] = [inferred_pos[0],
                                                        inferred_pos[1],
                                                        action2face[
                                                            dir2action[(inferred_pos[0] - cur_state["pos"][1][0],
                                                                        inferred_pos[1] - cur_state["pos"][1][1])]]]
                                miss_action = dir2action[inferred_lastmove]
                                action = dir2action[(
                                inferred_pos[0] - cur_state["pos"][1][0], inferred_pos[1] - cur_state["pos"][1][1])]
                            else:
                                action = dir2action[inferred_lastmove]
                                miss_action = dir2action[
                                    (
                                    inferred_pos[0] - cur_state["pos"][1][0], inferred_pos[1] - cur_state["pos"][1][1])]

                    else:
                        print("step {} missed more than 1 step".format(i + 1))
                        action = None
                    # return
                elif abs(dx) + abs(dz) > 1 and cur_state["hold"] != next_state["hold"]:
                    print("step {} missed move and interact".format(i + 1))
                    action = None
                elif abs(dx) + abs(dz) == 1:
                    if cur_state["hold"] == next_state["hold"]:
                        action = dir2action[(dx, dz)]
                    else:
                        # miss one interact
                        action = dir2action[(dx, dz)]
                        miss_state = copy.deepcopy(cur_state)
                        miss_state["pos"] = next_state["pos"]
                        miss_action = 'I'
                        print("step {} missed interact. inferred but may have error".format(i + 1))
                else:
                    action = dir2action[a]
            elif cur_state["hold"] != next_state["hold"]:
                action = "I"
            else:  # chop
                action = "C"
                # TODO test wait

            if miss_state:
                cur_state["action"] = action
                cur_state["reward"] = 0
                out_lines.append(cur_state)
                miss_state["action"] = miss_action
                miss_state["reward"] = next_state["score"] - cur_state["score"]
                out_lines.append(miss_state)
            else:
                cur_state["action"] = action
                cur_state["reward"] = next_state["score"] - cur_state["score"]
                out_lines.append(cur_state)

    with open(outfile, "w") as fo:
        for line in out_lines:
            fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    a = [0, 1, 2]
    b = [0, 1, 2]
    print(a == b)
    infer_acton("../data/traj1_1.json", "../data/traj1_1_infer.json", "1-1")
