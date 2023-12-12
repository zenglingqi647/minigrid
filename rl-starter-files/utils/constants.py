SKILL_MDL_PATH = [
    "storage/skill-model-v2/Goto-Finetune",
    "storage/skill-model-v2/Open",
    "storage/skill-model-v2/PickUp",
    "storage/skill-model-v2/Unlock-Finetune",
    # "storage/skill-model-v1-curriculum/PutNext"
]

COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
OBJECTS = ["ball", "box", "key"]

GO_TO = {i + j * 6: f"go to the {COLORS[i]} {OBJECTS[j]}" for j in range(len(OBJECTS)) for i in range(len(COLORS))}

OPEN = {18 + i: f"open the {COLORS[i]} door" for i in range(len(COLORS))}

PICK_UP = {
    24 + i + j * 6: f"pick up a {COLORS[i]} {OBJECTS[j]}" for j in range(len(OBJECTS)) for i in range(len(COLORS))
}

UNLOCK = {42 + i: f"open the {COLORS[i]} door" for i in range(len(COLORS))}

PUT_NEXT = {
    48 + i + j * 6 + k * 36 + l * 216: f"put the {COLORS[i]} {OBJECTS[k]} next to the {COLORS[j]} {OBJECTS[l]}"
    for k in range(len(OBJECTS)) for l in range(len(OBJECTS))
    for i in range(len(COLORS)) for j in range(len(COLORS)) if k != l
}

PUT_NEXT = f"put the {0} {1} next to the {2} {3}"

# In total, there are 18 + 6 + 18 + 6 + 6 * 3 * 6 * 3 = 366 configurations.

SKILL_LIST = [GO_TO, OPEN, PICK_UP, UNLOCK, PUT_NEXT]