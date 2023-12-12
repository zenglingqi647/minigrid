from __future__ import annotations

from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc
from minigrid.envs.babyai.goto import GoToRedBallGrey, GoToRedBall, GoToRedBlueBall


GOTO_DICT = {
    0: {
        "name": "GoToObj",
        # "envs": ["BabyAI-GoToObjS4-v0", "BabyAI-GoToObjS6-v0", "BabyAI-GoToObj-v0"]
        "envs": ["BabyAI-GoToObj-v0"]
    },
    1: {
        "name": "GoToDoor",
        "envs": ["BabyAI-GoToDoor-v0"]
    },
    2: {
        "name":
            "GoToRedBallGrey",
        "envs": [
            'BabyAI-GoToRedBallGreyS8N2', 'BabyAI-GoToRedBallGreyS8N3', 'BabyAI-GoToRedBallGreyS8N4',
            'BabyAI-GoToRedBallGreyS8N5', 'BabyAI-GoToRedBallGreyS8N6', "BabyAI-GoToRedBallGrey-v0"
        ]
    },
    3: {
        "name":
            "GoToRedBall",
        "envs": [
            'BabyAI-GoToRedBallS8N2', 'BabyAI-GoToRedBallS8N3', 'BabyAI-GoToRedBallS8N4', 'BabyAI-GoToRedBallS8N5',
            'BabyAI-GoToRedBallS8N6', "BabyAI-GoToRedBall-v0"
        ]
    },
    4: {
        "name":
            "GoToRedBlueBall",
        "envs": [
            'BabyAI-GoToRedBlueBallS8N2', 'BabyAI-GoToRedBlueBallS8N3', 'BabyAI-GoToRedBlueBallS8N4',
            'BabyAI-GoToRedBlueBallS8N5', 'BabyAI-GoToRedBlueBallS8N6', "BabyAI-GoToRedBlueBall-v0"
        ]
    },
    5: {
        "name":
            "CustomGoToObjDoor",
        "envs": [
            'BabyAI-CustomGoToObjDoorS8N2', 'BabyAI-CustomGoToObjDoorS8N3', 'BabyAI-CustomGoToObjDoorS8N4',
            'BabyAI-CustomGoToObjDoorS8N5', 'BabyAI-CustomGoToObjDoorS8N6', 'BabyAI-CustomGoToObjDoorS8N7',
            "BabyAI-GoToObjDoor-v0"
        ]
    },
    6: {
        "name":
            "GoToLocal",
        "envs": [
            "BabyAI-GoToLocalS5N2-v0", "BabyAI-GoToLocalS6N2-v0", "BabyAI-GoToLocalS6N3-v0", "BabyAI-GoToLocalS6N4-v0",
            "BabyAI-GoToLocalS7N4-v0", "BabyAI-GoToLocalS7N5-v0", "BabyAI-GoToLocalS8N2-v0", "BabyAI-GoToLocalS8N3-v0",
            "BabyAI-GoToLocalS8N4-v0", "BabyAI-GoToLocalS8N5-v0", "BabyAI-GoToLocalS8N6-v0", "BabyAI-GoToLocalS8N7-v0",
            "BabyAI-GoToLocal-v0"
        ]
    },
}


class GoToRedBallGreyS8N2(GoToRedBallGrey):

    def __init__(self):
        super().__init__(room_size=8, num_dists=2)


class GoToRedBallGreyS8N2(GoToRedBallGrey):

    def __init__(self):
        super().__init__(room_size=8, num_dists=2)


class GoToRedBallGreyS8N3(GoToRedBallGrey):

    def __init__(self):
        super().__init__(room_size=8, num_dists=3)


class GoToRedBallGreyS8N4(GoToRedBallGrey):

    def __init__(self):
        super().__init__(room_size=8, num_dists=4)


class GoToRedBallGreyS8N5(GoToRedBallGrey):

    def __init__(self):
        super().__init__(room_size=8, num_dists=5)


class GoToRedBallGreyS8N6(GoToRedBallGrey):

    def __init__(self):
        super().__init__(room_size=8, num_dists=6)


class GoToRedBallS8N2(GoToRedBall):

    def __init__(self):
        super().__init__(room_size=8, num_dists=2)


class GoToRedBallS8N3(GoToRedBall):

    def __init__(self):
        super().__init__(room_size=8, num_dists=3)


class GoToRedBallS8N4(GoToRedBall):

    def __init__(self):
        super().__init__(room_size=8, num_dists=4)


class GoToRedBallS8N5(GoToRedBall):

    def __init__(self):
        super().__init__(room_size=8, num_dists=5)


class GoToRedBallS8N6(GoToRedBall):

    def __init__(self):
        super().__init__(room_size=8, num_dists=6)


class GoToRedBlueBallS8N2(GoToRedBlueBall):

    def __init__(self):
        super().__init__(room_size=8, num_dists=2)


class GoToRedBlueBallS8N3(GoToRedBlueBall):

    def __init__(self):
        super().__init__(room_size=8, num_dists=3)


class GoToRedBlueBallS8N4(GoToRedBlueBall):

    def __init__(self):
        super().__init__(room_size=8, num_dists=4)


class GoToRedBlueBallS8N5(GoToRedBlueBall):

    def __init__(self):
        super().__init__(room_size=8, num_dists=5)


class GoToRedBlueBallS8N6(GoToRedBlueBall):

    def __init__(self):
        super().__init__(room_size=8, num_dists=6)


class CustomGoToObjDoor(RoomGridLevel):
    """

    ## Description

    Go to an object or door
    (of a given type and color, in the current room)

    ## Mission Space

    "go to the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box", "key" or "door".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent goes to the object or door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToObjDoor-v0`

    """

    def __init__(self, num_dists=8, **kwargs):
        self.num_dists = num_dists
        super().__init__(**kwargs)

    def gen_mission(self):
        self.place_agent(1, 1)
        objs = self.add_distractors(1, 1, num_distractors=self.num_dists, all_unique=False)

        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append(door)

        self.check_objs_reachable()

        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class CustomGoToObjDoorS8N2(CustomGoToObjDoor):

    def __init__(self):
        super().__init__(room_size=8, num_dists=2)


class CustomGoToObjDoorS8N3(CustomGoToObjDoor):

    def __init__(self):
        super().__init__(room_size=8, num_dists=3)


class CustomGoToObjDoorS8N4(CustomGoToObjDoor):

    def __init__(self):
        super().__init__(room_size=8, num_dists=4)


class CustomGoToObjDoorS8N5(CustomGoToObjDoor):

    def __init__(self):
        super().__init__(room_size=8, num_dists=5)


class CustomGoToObjDoorS8N6(CustomGoToObjDoor):

    def __init__(self):
        super().__init__(room_size=8, num_dists=6)


class CustomGoToObjDoorS8N7(CustomGoToObjDoor):

    def __init__(self):
        super().__init__(room_size=8, num_dists=7)
