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