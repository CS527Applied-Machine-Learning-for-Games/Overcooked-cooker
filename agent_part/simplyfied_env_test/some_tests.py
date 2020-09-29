from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, OvercookedGridworld, OvercookedState, ObjectState, SoupState, Recipe
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator, ONION_DISPENSER, TOMATO_DISPENSER, POT, DISH_DISPENSER, SERVING_LOC
from overcooked_ai_py.agents.agent import AgentGroup, AgentPair, GreedyHumanModel, FixedPlanAgent, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS, MotionPlanner
from overcooked_ai_py.utils import save_pickle, load_pickle, iterate_over_json_files_in_dir, load_from_json, save_as_json

START_ORDER_LIST = ["any"]
n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState


def test_one_player_env():
    mdp = OvercookedGridworld.from_layout_name("cramped_room_single")
    env = OvercookedEnv.from_mdp(mdp, horizon=12)
    a0 = FixedPlanAgent([stay, w, w, e, e, n, e, interact, w, n, interact])
    ag = AgentGroup(a0)
    trajectory, timestep, total_sparse, total_shaped = env.run_agents(ag, display=False)

    print(trajectory[0][0])

    print(env.lossless_state_encoding_mdp(env.state))
    # print(timestep)
    # print(total_sparse)
    # print(total_shaped)


def test_two_player_env():
    mdp = OvercookedGridworld.from_layout_name("tutorial_0")
    env = OvercookedEnv.from_mdp(mdp, horizon=12)
    a0 = FixedPlanAgent([stay, w, w, e, e, n, e, interact, w, n, interact])
    a1 = FixedPlanAgent([stay, w, w, e, e, n, e, interact, w, n, interact])
    ag = AgentGroup(a0, a1)
    trajectory, timestep, total_sparse, total_shaped = env.run_agents(ag, display=False)

    print(trajectory[0][0])

    print(env.lossless_state_encoding_mdp(env.state))
    # print(timestep)
    # print(total_sparse)
    # print(total_shaped)


if __name__ == '__main__':
    # test_one_player_env()
    test_two_player_env()
