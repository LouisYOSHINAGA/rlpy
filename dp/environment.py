from enum import Enum
import random
from typing import TypeAlias, Literal, Self


class State:
    def __init__(self, row: int =-1, column: int =-1) -> None:
        self.row: int = row
        self.column: int = column

    def __repr__(self) -> str:
        return f"<State: [{self.row}, {self.column}]>"

    def __hash__(self) -> int:
        return hash((self.row, self.column))

    def __eq__(self, other: Self) -> bool:
        return self.row == other.row and self.column == other.column

    def clone(self) -> Self:
        return self.__class__(self.row, self.column)


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


Cell: TypeAlias = Literal[-1, 0, 1, 9]
Grid: TypeAlias = list[list[Cell]]
Reward: TypeAlias = float  # {-1, -0.04, 0, 1}
Prob: TypeAlias = float  # [0, 1]
TransProbs: TypeAlias = dict[State, Prob]

ORDINALY_CELL: Cell = -1
DAMAGE_CELL: Cell = 0
REWARD_CELL: Cell = 1
BLOCK_CELL: Cell = 9


class Environment:
    def __init__(self, grid: Grid, move_prob: Prob =0.8) -> None:
        self.grid: Grid = grid
        self.agent_state = State()
        self.default_reward: Reward = -0.04
        self.move_prob: Prob = move_prob
        self.reset()

    @property
    def row_length(self) -> int:
        return len(self.grid)

    @property
    def column_length(self) -> int:
        return len(self.grid[0])

    @property
    def actions(self) -> list[Action]:
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self) -> list[State]:
        states: list[State] = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                if self.grid[row][column] != BLOCK_CELL:
                    states.append(State(row, column))
        return states

    def reset(self) -> State:
        self.agent_state = State(self.row_length-1, 0)  # lower left
        return self.agent_state

    def step(self, action: Action) -> tuple[State, Reward, bool]:
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state
        return self.agent_state, reward, done

    def transit(self, state: State, action: Action) -> tuple[State|None, Reward, bool]:
        transition_probs: TransProbs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, 0, True

        next_states: list[State] = []
        probs: list[Prob] = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state: State = random.choices(next_states, weights=probs)[0]
        reward, done = self.reward_func(next_state)
        return next_state, reward, done

    def can_action_at(self, state: State) -> bool:
        return self.grid[state.row][state.column] == 0

    def transit_func(self, state: State, action: Action) -> TransProbs:
        if not self.can_action_at(state):  # terminal cell
            return {}

        trainsition_probs: dict[State, Prob] = {}
        for a in self.actions:
            if a == action:  # forward
                prob = self.move_prob
            elif a == Action(-1 * action.value):  # backward
                prob = 0
            else:  # sides
                prob = (1 - self.move_prob) / 2

            next_state: State = self._move(state, a)
            if next_state not in trainsition_probs:
                trainsition_probs[next_state] = prob
            else:
                trainsition_probs[next_state] += prob
        return trainsition_probs

    def _move(self, state: State, action: Action) -> State:
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state: State = state.clone()
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state
        if self.grid[next_state.row][next_state.column] == BLOCK_CELL:
            next_state = state
        return next_state

    def reward_func(self, state: State) -> tuple[Reward, bool]:
        attribute: Cell = self.grid[state.row][state.column]
        if attribute == REWARD_CELL:
            reward = 1
            done = True
        elif attribute == ORDINALY_CELL:
            reward = -1
            done = True
        else:
            reward = self.default_reward
            done = False
        return reward, done