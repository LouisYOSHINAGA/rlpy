import random
from environment import *


class RandomStepAgent:
    def __init__(self, env) -> None:
        self.actions: list[Action] = env.actions

    def policy(self, state: State) -> Action:
        return random.choice(self.actions)


def main() -> None:
    grid: Grid = [
        [DAMAGE_CELL, DAMAGE_CELL, DAMAGE_CELL,   REWARD_CELL],
        [DAMAGE_CELL,  BLOCK_CELL, DAMAGE_CELL, ORDINALY_CELL],
        [DAMAGE_CELL, DAMAGE_CELL, DAMAGE_CELL,   DAMAGE_CELL]
    ]
    env = Environment(grid)
    agent = RandomStepAgent(env)

    for i in range(10):
        state = env.reset()
        total_reward: Reward = 0
        done: bool = False
        while not done:
            action: Action = agent.policy(state)
            state, reward, done = env.step(action)
            total_reward += reward
        print(f"Episode {i:02d}: Agent gets {total_reward: .2f} rewards.")


if __name__ == "__main__":
    main()