placement_phase = 0
attack_source_phase = 1
attack_target_phase = 2
fortify_source_phase = 3
fortify_target_phase = 4

skip_action = -1
verbose = False


def handle_fortify_source_phase(self, action):
    self.recurrence = False
    if self.phase != fortify_source_phase:
        raise "out of order phase handling found in call to handle_fortify_source_phase(self, action)"

    if self.agent_gets_card:
        get_card(self.agent.hand)
        self.agent_gets_card = False

    if action == skip_action:
        self.phase = placement_phase
        reward = 0

    # attempting an illegal fortify
    from_terr = self.territories[action]
    if action != skip_action and (from_terr.owner != self.agent or from_terr.troop_count < 2):
        self.phase = fortify_source_phase
        # reward = self.get_reward(illegal=True)
        return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}

    self.from_terr = from_terr
    self.prev_move = action
    self.phase = fortify_target_phase
    reward = 0

    if self.phase == placement_phase:
        state, new_reward, terminated, truncated, info = self.handle_other_players()
    else:
        state, new_reward, terminated, truncated, info = self.get_state(), 0, self.agent_game_ended, False, {}
    return state, reward + new_reward, terminated, truncated, info


def handle_fortify_target_phase(self, action):
    self.recurrence = False
    if self.phase != fortify_target_phase:
        raise "out of order phase handling found in call to handle_fortify_target_phase(self, action)"

    to_terr = self.territories[action]
    # attempting an illegal fortify
    if (action == skip_action or to_terr == self.from_terr or to_terr.owner != self.agent or to_terr not in fortify_bfs(self.from_terr)):
        self.phase = fortify_source_phase
        self.from_terr = None
        reward = self.get_reward(illegal=True)
    else:
        troops = self.from_terr.troop_count - 1
        fortify(self.from_terr, to_terr, troops)
        self.from_terr = None
        self.phase = placement_phase
        reward = 0

    new_reward = 0
    if self.phase == placement_phase:
        state, new_reward, terminated, truncated, info = self.handle_other_players()
    else:
        state, new_reward, terminated, truncated, info = self.get_state(), 0, self.agent_game_ended, False, {}
    return state, reward + new_reward, terminated, truncated, info


def handle_fortify_phase(self, action):
    self.recurrence = False

    if self.phase != fortify_target_phase and self.phase != fortify_source_phase:
        raise "out of order phase handling"

    if self.agent_gets_card:
        get_card(self.agent.hand)
        self.agent_gets_card = False

    if action == skip_action:
        self.phase = placement_phase
        reward = 0
        if self.phasic_credit_assignment: reward = self.get_reward()

    if self.phase == fortify_source_phase:
        from_terr = self.territories[action]

        # attempting an illegal fortify
        if (from_terr.owner != self.agent or from_terr.troop_count < 2):
            self.phase = fortify_source_phase
            # reward = self.get_reward(illegal=True)
            return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}

        self.from_terr = from_terr
        self.prev_move = action
        self.phase = fortify_target_phase
        reward = 0
        if self.phasic_credit_assignment: reward = self.get_reward()

    elif self.phase == fortify_target_phase:
        to_terr = self.territories[action]
        # attempting an illegal fortify
        if (to_terr == self.from_terr or to_terr.owner != self.agent or to_terr not in fortify_bfs(self.from_terr)):
            self.phase = fortify_source_phase
            self.from_terr = None
            reward = self.get_reward(illegal=True)
        else:
            troops = self.from_terr.troop_count - 1
            fortify(self.from_terr, to_terr, troops)
            self.from_terr = None
            self.phase = placement_phase
            reward = 0

    if self.phase == placement_phase:
        state, new_reward, terminated, truncated, info = self.handle_other_players()
    else:
        state, new_reward, terminated, truncated, info = self.get_state(), 0, self.agent_game_ended, False, {}
    return state, reward + new_reward, terminated, truncated, info



