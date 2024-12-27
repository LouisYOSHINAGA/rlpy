from typing import TypeAlias, Literal

State: TypeAlias = str  # {"state_..."|"happy_end"|"bad_end"}
Action: TypeAlias = Literal["up", "down"]
Reward: TypeAlias = Literal[-1, 0, 1]
Prob: TypeAlias = float  # [0, 1]
TransProbs: TypeAlias = dict[State, Prob]
StateValue: TypeAlias = float

all_action: list[Action] = ["up", "down"]


def V(s: State, gamma: float =0.99) -> StateValue:
    return R(s) + gamma * max_V_on_next_state(s)

def R(s: State) -> Reward:
    if s == "happy_end":
        return 1
    elif s == "bad_end":
        return -1
    else:  # "state_..."
        return 0

def max_V_on_next_state(s: State) -> float:
    if s in ["happy_end", "bad_end"]:
        return 0

    Tvs: list[float] = []  # T(s'|s,a) V(s') forall s'
    for a in all_action:
        trans_probs: TransProbs = transit_func(s, a)  # T(s'|s,a)
        Tv: float = 0  # \sum_{s'} T(s'|s) V(s')
        for next_state, T in trans_probs.items():
            Tv += T * V(next_state)
        Tvs.append(Tv)
    return max(Tvs)  # max_{a} \sum_{s'} T(s'|s,a) V(s')

def transit_func(s: State, a: Action) -> TransProbs:
    LIMIT_GAME_COUNT: int = 5
    HAPPY_END_BORDER: int = 4
    MOVE_PROB: Prob = 0.9

    action_history: list[Action] = s.split("_")[1:]  # type: ignore
    if len(action_history) == LIMIT_GAME_COUNT:
        n_up: int = sum([a == "up" for a in action_history])
        state: State = "happy_end" if n_up >= HAPPY_END_BORDER else "bad_end"
        return {state: 1.0}

    opposite: Action = "up" if a == "down" else "down"
    return {
        f"{s}_{a}": MOVE_PROB,
        f"{s}_{opposite}": 1 - MOVE_PROB
    }


if __name__ == "__main__":
    print(f"{V("state")}")
    print(f"{V("state_up_up")}")
    print(f"{V("state_down_down")}")