from environment import State, Action, Reward, Prob, TransProbs
from typing import TypeAlias, Any, Generator

Env: TypeAlias = Any
StateValue: TypeAlias = float
StateValueFn: TypeAlias = dict[State, StateValue]
StateValueList2d: TypeAlias = list[list[StateValue]]


class Planner:
    def __init__(self, env: Env) -> None:
        self.env: Env = env
        self.log: list[StateValueList2d] = []

    def initialize(self) -> None:
        self.env.reset()
        self.log = []

    def plan(self, gamma: float =0.9, threshold: float =0.0001) -> StateValueList2d:
        raise NotImplementedError()

    def transitions_at(self, state: State, action: Action) -> Generator[tuple[Prob, State, Reward], None, None]:
        trans_probs: TransProbs = self.env.transit_func(state, action)  # T(s'|s,a)
        for s_, T in trans_probs.items():
            R, _ = self.env.reward_func(s_)  # R(s')
            yield T, s_, R  # T(s'|s,a), s', R(s')

    def format_state_value_fn(self, V: StateValueFn) -> StateValueList2d:
        Vlist: StateValueList2d = [
            [0 for _ in range(self.env.column_length)] for _ in range(self.env.row_length)
        ]
        for state, state_value in V.items():
            Vlist[state.row][state.column] = state_value
        return Vlist 


class ValueIterationPlanner(Planner):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

    # estimate state value function: V(s)
    def plan(self, gamma: float =0.9, threshold: float =0.001) -> StateValueList2d:
        self.initialize()
        V: StateValueFn = {s: 0 for s in self.env.states}

        while True:
            delta: float = 0
            self.log.append(self.format_state_value_fn(V))

            for s in V.keys():
                if not self.env.can_action_at(s):
                    continue

                cur_V: Reward = V[s]  # V_{i}(s)
                v_cands: list[Reward] = []  # sum_{s'} T(s'|s,a) * ( R(s) + gamma * V(s') )
                for a in self.env.actions:
                    v: Reward = 0
                    for T, s_, R in self.transitions_at(s, a):
                        v += T * (R + gamma * V[s_])  # T(s'|s,a) * ( R(s') + gamma * V(s') )
                    v_cands.append(v)
                V[s] = max(v_cands)  # V_{i+1}(s) = max_{a} sum_{s'} T(s'|s,a) * ( R(s) + gamma * V_{i}(s') )
                delta = max(delta, abs(V[s] - cur_V))  # max | V_{i+1}(s) - V_{i}(s) |

            if delta < threshold:
                break

        return self.format_state_value_fn(V)


class PolicyIterationPlanner(Planner):
    ...