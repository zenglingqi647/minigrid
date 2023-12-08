from .format import Vocabulary

def interact_with_human():
    skill_num = int(input("What skill should the model use?"))
    goal_text = input("What should the goal of the skill be?")
    return skill_num, goal_text