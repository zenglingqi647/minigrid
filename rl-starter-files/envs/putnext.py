"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission, with the `Put Next` instruction.
"""
from __future__ import annotations

from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, PutNextInstr

putnext_dict = {
    0: {
        "name": "PutNextLocal",
        "envs": [f'BabyAI-PutNextLocalS{s}N{n}' for s in range(5, 9) for n in range(2, 5)]
    },
}


class CustomPutNextLocal(RoomGridLevel):
    """

    ## Description

    Put an object next to another object, inside a single room
    with no doors, no distractors

    ## Mission Space

    "put the {color} {type} next to the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

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

    1. The agent finishes the instructed task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-PutNextLocal-v0`
    - `BabyAI-PutNextLocalS5N3-v0`
    - `BabyAI-PutNextLocalS6N4-v0``

    """

    def __init__(self, room_size=8, num_objs=8, **kwargs):
        self.num_objs = num_objs
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(ObjDesc(o1.type, o1.color), ObjDesc(o2.type, o2.color))
