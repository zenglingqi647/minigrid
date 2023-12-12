import re
from utils.constants import *

SKILL_LIST = [GO_TO, OPEN, PICK_UP, UNLOCK, PUT_NEXT]
COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
OBJECTS = ["ball", "box", "key"]


def parse_goal(skill, goal):
    color_pattern = "(" + "|".join(COLORS) + ")"
    object_pattern = "(" + "|".join(OBJECTS) + ")"

    goal = goal.lower()  # Convert to lowercase for easier matching

    if skill == 0:  # Go to Object
        match = re.search(f"go to (?:a|the) {color_pattern} {object_pattern}", goal)
        if match:
            return {'skill': 0, 'color': match.group(1), 'type': match.group(2)}

    elif skill == 1:  # Open door
        match = re.search(f"open (?:a|the) {color_pattern} door", goal)
        if match:
            return {'skill': 1, 'color': match.group(1), 'type': 'door'}

    elif skill == 2:  # Pickup an item
        match = re.search(f"pick up (?:a|the) {color_pattern} {object_pattern}", goal)
        if match:
            return {'skill': 2, 'color': match.group(1), 'type': match.group(2)}

    elif skill == 3:  # Unlock a door
        match = re.search(f"unlock (?:a|the) {color_pattern} door", goal)
        if match:
            return {'skill': 3, 'color': match.group(1), 'type': 'door'}

    elif skill == 4:  # Put next to
        # put the {color} {type} next to the {color} {type}
        match = re.search(
            f"put (?:a|the) {color_pattern} {object_pattern} next to (?:a|the) {color_pattern} {object_pattern}", goal)
        if match:
            return {
                'skill': 4,
                'color1': match.group(1),
                'type1': match.group(2),
                'color2': match.group(3),
                'type2': match.group(4)
            }

    return None  # Return None if no match is found


def validate_goal(skill, goal):
    if not parse_goal(skill, goal):
        raise ValueError('Goal and skill mismatch!')


def similarity(llm_skill, llm_goal, dqn_skill, dqn_goal):
    llm_return = parse_goal(llm_skill, llm_goal)
    dqn_return = parse_goal(dqn_skill, dqn_goal)

    if not llm_return:
        raise ValueError('LLM: Goal and skill mismatch!')
    if not dqn_return:
        raise ValueError('DQN: Goal and skill mismatch!')

    # compare the skill, colors and types
    similarity = []
    for key in llm_return.keys():
        similarity.append(llm_return[key] == dqn_return[key])
    return sum(similarity) / len(similarity)


# TODO: validation, similarity
if __name__ == "__main__":
    skill = 0
    goal = "Go to the red ball"
    parsed_goal = parse_goal(skill, goal)
    print(parsed_goal)

    skill = 1
    goal = "Go to the black ball"
    parsed_goal = parse_goal(skill, goal)
    print(parsed_goal)

    skill = 4
    goal = "put the red ball next to the green box"
    parsed_goal = parse_goal(skill, goal)
    print(parsed_goal)

    skill = 1
    goal = "Go to the red ball"
    parsed_goal = parse_goal(skill, goal)
    print(parsed_goal)

    skill = 1
    goal = "Go to the red goal"
    parsed_goal = parse_goal(skill, goal)
    print(parsed_goal)

    print(similarity(0, "Go to the red ball", 0, "Go to the red ball"), 1)
    print(similarity(0, "Go to the red ball", 0, "Go to the green ball"), 0.67)
    print(similarity(0, "Go to the red ball", 1, "Go to the red box"), 'error')
    print(similarity(0, "Go to the red ball", 0, "Go to the red box"), 0)