from gymnasium.envs.registration import register
from gymnasium.envs.registration import registry


# Unregister the original BabyAI environments
def unregister_env(env_id):
    if env_id in registry:
        del registry[env_id]


# GoTo
env_types = ["GoToRedBallGrey", "GoToRedBall", "GoToRedBlueBall", "CustomGoToObjDoor"]

for env_type in env_types:
    for n in range(2, 8 if env_type == "CustomGoToObjDoor" else 7):
        env_id = f'BabyAI-{env_type}S8N{n}'
        entry_point = f'envs.goto:{env_type}S8N{n}'
        register(id=env_id, entry_point=entry_point)

# PickUp
for n in range(1, 9):
    register(id=f'BabyAI-CustomPickupLocN{n}', entry_point=f'envs.pickup:CustomPickupLocN{n}')

# PutNext
unregister_env('BabyAI-PutNextLocalS5N3-v0')
unregister_env('BabyAI-PutNextLocalS6N4-v0')
for s in range(5, 9):
    for n in range(2, 5):
        register(
            id=f'BabyAI-CustomPutNextLocalS{s}N{n}',
            entry_point=f'envs.putnext:CustomPutNextLocal',
            kwargs={
                "room_size": s,
                "num_objs": n
            },
        )

for n in range(1, 3):
    register(id=f'BabyAI-CustomUnlockLocalN{n}', entry_point=f'envs.unlock:CustomUnlockLocalN{n}')