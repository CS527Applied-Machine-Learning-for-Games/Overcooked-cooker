def traj2demo(trajectories):
    trajectories["ep_observations"] = trajectories.pop("ep_states")
    for ep_list in trajectories["ep_observations"]:
        for step_dicts in ep_list:
            step_dicts["order_list"] = None
            for obj in step_dicts["objects"]:
                if obj["name"] == "soup":
                    obj["state"] = [obj["_ingredients"][0]["name"],
                                    len(obj["_ingredients"]),
                                    max([0, obj["cooking_tick"]])]
            for player_dict in step_dicts["players"]:
                held_obj = player_dict["held_object"]
                if held_obj is not None and held_obj["name"] == "soup":
                    held_obj["state"] = [held_obj["_ingredients"][0]["name"],
                                         len(held_obj["_ingredients"]),
                                         held_obj["cooking_tick"]]


    for ep_list in trajectories["ep_actions"]:
        for i in range(len(ep_list)):
            ep_list[i] = list(ep_list[i])
            for j in range(len(ep_list[i])):
                if isinstance(ep_list[i][j], str):
                    ep_list[i][j] = 'INTERACT'

    for ep_params in trajectories["mdp_params"]:
        extra_info = {
            "num_items_for_soup": 3,
            "rew_shaping_params": None,
            "cook_time": 20,
            "start_order_list": None
        }
        ep_params.update(extra_info)

    return trajectories


def demo2traj(filename):
    traj_dict = load_from_json(filename)
    for ep_list in traj_dict["ep_observations"]:
        for step_dict in ep_list:
            for state_dict in step_dict["players"]:
                if "held_object" not in state_dict:
                    state_dict["held_object"] = None

    traj_dict["ep_states"] = [[OvercookedState.from_dict(ob) for ob in curr_ep_obs] for curr_ep_obs in
                              traj_dict["ep_observations"]]
    traj_dict["ep_actions"] = [[tuple(tuple(a) if type(a) is list else "interact" for a in j_a) for j_a in ep_acts] for ep_acts
                               in traj_dict["ep_actions"]]
    return traj_dict