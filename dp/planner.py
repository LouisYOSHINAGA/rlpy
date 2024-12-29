from environment import Environment, State, Action, Reward, Prob, TransProbs
from typing import TypeAlias, Generator

StateValue: TypeAlias = float
StateValueFn: TypeAlias = dict[State, StateValue]
StateValueList2d: TypeAlias = list[list[StateValue]]
ActionValue: TypeAlias = float
ActionValueFn: TypeAlias = dict[tuple[State, Action], ActionValue]
Policy: TypeAlias = dict[State, dict[Action, Prob]]


class Planner:
    def __init__(self, env: Environment) -> None:
        self.env: Environment = env
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

    def format_V(self, V: StateValueFn) -> StateValueList2d:
        Vlist: StateValueList2d = [
            [0 for _ in range(self.env.column_length)] for _ in range(self.env.row_length)
        ]
        for state, state_value in V.items():
            Vlist[state.row][state.column] = state_value
        return Vlist


class ValueIterationPlanner(Planner):
    def __init__(self, env: Environment) -> None:
        super().__init__(env)

    # estimate state value function V(s)
    def plan(self, gamma: float =0.9, threshold: float =0.001) -> StateValueList2d:
        self.initialize()
        V: StateValueFn = {s: 0 for s in self.env.states}

        while True:
            delta: float = 0
            self.log.append(self.format_V(V))

            for s in V.keys():
                if not self.env.can_action_at(s):
                    continue

                cur_V: StateValue = V[s]  # V_{i}(s)
                v_cands: list[StateValue] = []  # sum_{s'} T(s'|s,a) * ( R(s) + gamma * V(s') )
                for a in self.env.actions:
                    v: StateValue = 0
                    for T, s_, R in self.transitions_at(s, a):
                        v += T * (R + gamma * V[s_])  # T(s'|s,a) * ( R(s') + gamma * V(s') )
                    v_cands.append(v)
                V[s] = max(v_cands)  # V_{i+1}(s) = max_{a} sum_{s'} T(s'|s,a) * ( R(s) + gamma * V_{i}(s') )

                delta = max(delta, abs(V[s] - cur_V))  # max | V_{i+1}(s) - V_{i}(s) |
            if delta < threshold:
                break

        return self.format_V(V)


class PolicyIterationPlanner(Planner):
    def __init__(self, env: Environment) -> None:
        super().__init__(env)
        self.policy: Policy = {}

    def initialize(self) -> None:
        super().initialize()
        self.policy = {}

        for s in self.env.states:
            self.policy[s] = {}
            for a in self.env.actions:
                self.policy[s][a] = 1 / len(self.env.actions)

    # estimate state value funtion V(s) from policy pi(a|s)
    def estimate_V_from_pi(self, gamma: float, threshold: float) -> StateValueFn:
        V: StateValueFn = {s: 0 for s in self.env.states}

        while True:
            delta: float = 0

            for s in V.keys():
                cur_V: StateValue = V[s]  # V_{i}(s)
                v_cands: list[StateValue] = []  # sum_{s'} T(s'|s,a) * ( R(s) + gamma * V(s') )
                for a, pi in self.policy[s].items():
                    v: StateValue = 0
                    for T, s_, R in self.transitions_at(s, a):
                        v += pi * T * (R  + gamma * V[s_])  # pi(a|s) * T(s'|s,a) * ( R(s) + gamma * V(s') )
                    v_cands.append(v)
                V[s] = sum(v_cands)  # V_{i+1}(s) = sum_{a} pi(a|s) sum_{s'} T(s'|s,a) * ( R(s) + gamma * V_{i}(s') )

                delta = max(delta, abs(V[s] - cur_V))  # max | V_{i+1}(s) - V_{i}(s) |
            if delta < threshold:
                break

        return V

    # estimate state value function V(s) with action value function Q(a)
    def plan(self, gamma: float =0.9, threshold: float =0.001) -> StateValueList2d:
        self.initialize()

        while True:
            V: StateValueFn = self.estimate_V_from_pi(gamma, threshold)
            self.log.append(self.format_V(V))

            # estimate action value function Q(a)
            update_stable: bool = True

            for s in self.env.states:
                Q: ActionValueFn = {(s, a): 0 for a in self.env.actions}
                for a in self.env.actions:
                    for T, s_, R in self.transitions_at(s, a):
                        Q[s, a] += T * (R + gamma * V[s_])  # T(s'|s,a) * ( R(s) + gamma * V(s') )

                best_action: Action = max(Q, key=Q.get)[1]  # type: ignore  # argmax_{a} Q(s,a)
                for a in self.policy[s]:
                    self.policy[s][a] = int(a == best_action)

                max_policy_action: Action = max(self.policy[s], key=self.policy[s].get)  # type: ignore
                update_stable |= (max_policy_action == best_action)
            if update_stable:
                break

        return self.format_V(V)