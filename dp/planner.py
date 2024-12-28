from typing import TypeAlias, Any, Generator

State: TypeAlias = Any
Action: TypeAlias = Any
Env: TypeAlias = Any
Prob: TypeAlias = float  # [0, 1)
TransProbs: TypeAlias = dict[State, Prob]
Reward: TypeAlias = float
Grid: TypeAlias = list[list[Reward]]


class Planner:
    def __init__(self, env: Env) -> None:
        self.env: Env = env
        self.log: list[Grid] = []

    def initialize(self) -> None:
        self.env.reseet()
        self.log = []

    def plan(self, gamma: float =0.9, threshold: float =0.0001) -> Grid:
        raise NotImplementedError()

    def transitions_at(self, state: State, action: Action) -> Generator[tuple[Prob, State, Reward], None, None]:
        trans_probs: TransProbs = self.env.transit_func(state, action)  # T(s'|s,a)
        for next_state, T in trans_probs.items():
            reward, _ = self.env.reward_func(next_state)  # R(s')
            yield T, next_state, reward

    def dict_to_grid(self, state_reward: dict[State, Reward]) -> Grid:
        grid: Grid = [
            [0 for _ in range(self.env.column_length)] for _ in range(self.env.row_length)
        ]
        for state, reward in state_reward.items():
            grid[state.row][state.column] = reward
        return grid


class ValueIterationPlanner(Planner):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

    def plan(self, gamma: float =0.9, threshold: float =0.001) -> Grid:
        self.initialize()
        actions: list[Action] = self.env.actions
        V: dict[State, Reward] = {s: 0 for s in self.env.states}

        while True:
            delta: float = 0
            self.log.append(self.dict_to_grid(V))
            for s in V.keys():
                if not self.env.can_action_at(s):
                    continue
                expected_rewards: list[Reward] = []
                for a in actions:
                    r: Reward = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                max_reward: Reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            if delta < threshold:
                break

        return self.dict_to_grid(V)