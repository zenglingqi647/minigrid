# goto
'''
GoToObj
'''

envs = [
    "BabyAI-Open-v0", "BabyAI-OpenRedDoor-v0", "BabyAI-OpenDoor-v0", "BabyAI-OpenDoorDebug-v0",
    "BabyAI-OpenDoorColor-v0", "BabyAI-OpenDoorLoc-v0", "BabyAI-OpenTwoDoors-v0", "BabyAI-OpenRedBlueDoors-v0",
    "BabyAI-OpenRedBlueDoorsDebug-v0", "BabyAI-OpenDoorsOrderN2-v0", "BabyAI-OpenDoorsOrderN4-v0",
    "BabyAI-OpenDoorsOrderN2Debug-v0", "BabyAI-OpenDoorsOrderN4Debug-v0", "BabyAI-Pickup-v0", "BabyAI-UnblockPickup-v0",
    "BabyAI-PickupLoc-v0", "BabyAI-PickupDist-v0", "BabyAI-PickupDistDebug-v0", "BabyAI-PickupAbove-v0",
    "BabyAI-PutNextLocal-v0", "BabyAI-PutNextLocalS5N3-v0", "BabyAI-PutNextLocalS6N4-v0", "BabyAI-PutNextS4N1-v0",
    "BabyAI-PutNextS5N2-v0", "BabyAI-PutNextS5N1-v0", "BabyAI-PutNextS6N3-v0", "BabyAI-PutNextS7N4-v0",
    "BabyAI-PutNextS5N2Carrying-v0", "BabyAI-PutNextS6N3Carrying-v0", "BabyAI-PutNextS7N4Carrying-v0",
    "BabyAI-Unlock-v0", "BabyAI-UnlockLocal-v0", "BabyAI-UnlockLocalDist-v0", "BabyAI-KeyInBox-v0",
    "BabyAI-UnlockPickup-v0", "BabyAI-UnlockPickupDist-v0", "BabyAI-BlockedUnlockPickup-v0", "BabyAI-UnlockToUnlock-v0",
    "BabyAI-ActionObjDoor-v0", "BabyAI-FindObjS5-v0", "BabyAI-FindObjS6-v0", "BabyAI-FindObjS7-v0",
    "BabyAI-KeyCorridor-v0", "BabyAI-KeyCorridorS3R1-v0", "BabyAI-KeyCorridorS3R2-v0", "BabyAI-KeyCorridorS3R3-v0",
    "BabyAI-KeyCorridorS4R3-v0", "BabyAI-KeyCorridorS5R3-v0", "BabyAI-KeyCorridorS6R3-v0", "BabyAI-OneRoomS8-v0",
    "BabyAI-OneRoomS12-v0", "BabyAI-OneRoomS16-v0", "BabyAI-OneRoomS20-v0", "BabyAI-MoveTwoAcrossS5N2-v0",
    "BabyAI-MoveTwoAcrossS8N9-v0", "BabyAI-MiniBossLevel-v0", "BabyAI-BossLevel-v0", "BabyAI-BossLevelNoUnlock-v0"
]


class Curriculum:

    def __init__(self, env_dict, upgrade_threshold=0.6, downgrade_threshold=0.3, repeat_threshold=5):
        """Curriculum class for selecting environments and updating the difficulty.

        Args:
            env_dict (int->dict[name, envs]): Dictionary of levels and their environments.
            upgrade_threshold (float, optional): Defaults to 0.6.
            downgrade_threshold (float, optional): Defaults to 0.3.
        """
        self.env_dict = env_dict
        self.current_level = 0
        # thresholds
        self.upgrade_threshold = upgrade_threshold
        self.downgrade_threshold = downgrade_threshold
        self.repeated_threshold = repeat_threshold
        # env difficulty of current level
        self.env_idx = 0
        self.repeated = 0
        self.finished_levels = []

    def select_environment(self):
        """
        Selects an environment based on the current level.
        """
        envs_at_level = self.env_dict[self.current_level]['envs']
        return envs_at_level[self.env_idx]

    def update_level(self, success_rate):
        """
        Updates the level and env difficulties based on the success rate.
        """
        # update difficulty
        if success_rate > self.upgrade_threshold and self.env_idx < len(self.env_dict[self.current_level]['envs']):
            self.env_idx += 1
        elif success_rate < self.downgrade_threshold and self.env_idx > 0:
            self.env_idx -= 1
        else:
            self.repeated += 1

        # update level
        if self.repeated > self.repeated_threshold:
            self.repeated = 0
            self.env_idx = 0
            self.current_level += 1
            self.current_level = self.current_level % len(self.env_dict.keys())

        if self.env_idx >= len(self.env_dict[self.current_level]['envs']):
            self.finished_levels.append(self.current_level)
            self.repeated = 0
            self.env_idx = 0
            self.current_level += 1
            self.current_level = self.current_level % len(self.env_dict.keys())

        # check if level is finished
        while self.current_level in self.finished_levels:
            self.current_level += 1


if __name__ == '__main__':
    from envs.goto import goto_dict
    curriculum = Curriculum(env_dict=goto_dict)
    for i in range(10):
        print(curriculum.select_environment())
        curriculum.update_level(0.5)
    print('-------------------')
    for i in range(10):
        print(curriculum.select_environment())
        curriculum.update_level(0.7)
    print('-------------------')
    for i in range(10):
        print(curriculum.select_environment())
        curriculum.update_level(0.1)