import json

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


def infer_acton(infile, outfile):
    # only for chef 1
    with open(infile) as fi:
        in_lines = list(map(json.loads, fi.readlines()))
        for i in range(len(in_lines) - 1):
            cur_state = in_lines[i]
            next_state = in_lines[i + 1]
            if cur_state["pos"] != next_state["pos"]:
                dx = next_state["pos"][1][0] - cur_state["pos"][1][0]
                dz = next_state["pos"][1][1] - cur_state["pos"][1][1]
                a = next_state["pos"][1][2]
                if abs(dx) + abs(dz) > 1:
                    print("step {} missed".format(i + 1))
                    action = None
                    # return
                elif abs(dx) + abs(dz) == 1:
                    action = dir2action[(dx, dz)]
                else:
                    action = dir2action[a]
            elif cur_state["hold"] != next_state["hold"]:
                action = "I"
            else:  # chop
                action = "C"

            reward = next_state["score"] - cur_state["score"]
            cur_state["action"] = action
            cur_state["reward"] = reward

    with open(outfile, "w") as fo:
        for line in in_lines:
            fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    a = [0, 1, 2]
    b = [0, 1, 2]
    print(a == b)
    infer_acton("../data/test.json", "../data/test_infer.json")
