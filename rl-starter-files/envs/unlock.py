from __future__ import annotations

from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, OpenInstr

unlock_dict = {
    0: {
        "name":
            "UnlockLocal",
        "envs": [
            f"BabyAI-UnlockLocal-v0", "BabyAI-CustomUnlockLocalN1", "BabyAI-CustomUnlockLocalN2",
            "BabyAI-UnlockLocalDist-v0"
        ]
    },
    1: {
        "name": "KeyInBox",
        "envs": ["BabyAI-KeyInBox-v0"]
    },
}


class CustomUnlockLocal(RoomGridLevel):
    """

    ## Description

    Fetch a key and unlock a door
    (in the current room)

    ## Mission Space

    "open the door"

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

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-UnlockLocal-v0`
    - `BabyAI-UnlockLocalDist-v0`

    """

    def __init__(self, distractors=None, **kwargs):
        self.distractors = distractors
        super().__init__(**kwargs)

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)
        self.add_object(1, 1, "key", door.color)
        if self.distractors:
            self.add_distractors(1, 1, num_distractors=self.distractors)
        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class CustomUnlockLocalN1(CustomUnlockLocal):

    def __init__(self):
        super().__init__(distractors=1)


class CustomUnlockLocalN2(CustomUnlockLocal):

    def __init__(self):
        super().__init__(distractors=2)
