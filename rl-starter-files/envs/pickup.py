from __future__ import annotations

from minigrid.envs.babyai.core.levelgen import LevelGen

pickup_dict = {
    0: {
        "name": "CustomPickupLoc",
        "envs": [f"BabyAI-CustomPickupLocN{n}" for n in range(1, 9)]
    },
    1: {
        "name": "PickupDist",
        "envs": ["BabyAI-PickupDist-v0"]
    },
    2: {
        "name": "PickupAbove",
        "envs": ["BabyAI-PickupAbove-v0"]
    },
}


class CustomPickupLoc(LevelGen):
    """

    ## Description

    Pick up an object which may be described using its location. This is a
    single room environment.

    Competencies: PickUp, Loc. No unblocking.

    ## Mission Space

    "pick up the {color} {type}"

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

    1. The agent picks up the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-PickupLoc-v0`

    """

    def __init__(self, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(**kwargs)


for n in range(1, 8):
    class_name = f"CustomPickupLocN{n}"
    class_body = type(
        class_name, (CustomPickupLoc,), {
            "__init__":
                lambda self, **kwargs: CustomPickupLoc.__init__(self,
                                                                action_kinds=["pickup"],
                                                                instr_kinds=["action"],
                                                                num_rows=n,
                                                                num_cols=n,
                                                                num_dists=n,
                                                                locked_room_prob=0,
                                                                locations=True,
                                                                unblocking=False,
                                                                **kwargs)
        })
    globals()[class_name] = class_body