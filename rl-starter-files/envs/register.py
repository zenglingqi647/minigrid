from gymnasium.envs.registration import register

# GoTo
register(id='BabyAI-GoToRedBallGreyS8N2', entry_point='envs.goto:GoToRedBallGreyS8N2')
register(id='BabyAI-GoToRedBallGreyS8N3', entry_point='envs.goto:GoToRedBallGreyS8N3')
register(id='BabyAI-GoToRedBallGreyS8N4', entry_point='envs.goto:GoToRedBallGreyS8N4')
register(id='BabyAI-GoToRedBallGreyS8N5', entry_point='envs.goto:GoToRedBallGreyS8N5')
register(id='BabyAI-GoToRedBallGreyS8N6', entry_point='envs.goto:GoToRedBallGreyS8N6')

register(id='BabyAI-GoToRedBallS8N2', entry_point='envs.goto:GoToRedBallS8N2')
register(id='BabyAI-GoToRedBallS8N3', entry_point='envs.goto:GoToRedBallS8N3')
register(id='BabyAI-GoToRedBallS8N4', entry_point='envs.goto:GoToRedBallS8N4')
register(id='BabyAI-GoToRedBallS8N5', entry_point='envs.goto:GoToRedBallS8N5')
register(id='BabyAI-GoToRedBallS8N6', entry_point='envs.goto:GoToRedBallS8N6')

register(id='BabyAI-GoToRedBlueBallS8N2', entry_point='envs.goto:GoToRedBlueBallS8N2')
register(id='BabyAI-GoToRedBlueBallS8N3', entry_point='envs.goto:GoToRedBlueBallS8N3')
register(id='BabyAI-GoToRedBlueBallS8N4', entry_point='envs.goto:GoToRedBlueBallS8N4')
register(id='BabyAI-GoToRedBlueBallS8N5', entry_point='envs.goto:GoToRedBlueBallS8N5')
register(id='BabyAI-GoToRedBlueBallS8N6', entry_point='envs.goto:GoToRedBlueBallS8N6')

register(id='BabyAI-CustomGoToObjDoorS8N2', entry_point='envs.goto:CustomGoToObjDoorS8N2')
register(id='BabyAI-CustomGoToObjDoorS8N3', entry_point='envs.goto:CustomGoToObjDoorS8N3')
register(id='BabyAI-CustomGoToObjDoorS8N4', entry_point='envs.goto:CustomGoToObjDoorS8N4')
register(id='BabyAI-CustomGoToObjDoorS8N5', entry_point='envs.goto:CustomGoToObjDoorS8N5')
register(id='BabyAI-CustomGoToObjDoorS8N6', entry_point='envs.goto:CustomGoToObjDoorS8N6')
register(id='BabyAI-CustomGoToObjDoorS8N7', entry_point='envs.goto:CustomGoToObjDoorS8N7')

# PickUp
for n in range(1, 9):
    register(id=f'BabyAI-CustomPickupLocN{n}', entry_point=f'envs.pickup:CustomPickupLocN{n}')

# PutNext
for s in range(5, 9):
    for n in range(2, 5):
        register(
            id=f'BabyAI-PutNextLocalS{s}N{n}',
            entry_point=f'envs.putnext:PutNextLocal',
            kwargs={
                "room_size": s,
                "num_objs": n
            },
        )
