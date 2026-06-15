"""Evaluate exploitability of a trained PPO agent in Kuhn Poker."""

import numpy as np
import torch
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability
from open_spiel.python.pytorch.ppo import PPO, PPOAgent


# ── wrap PPO agent as an OpenSpiel Policy ────────────────────────────────────

class PPOPolicy(policy_lib.Policy):
    """Wraps a trained PPO agent into OpenSpiel's Policy interface."""

    def __init__(self, game, agent, player_id):
        super().__init__(game, list(range(game.num_players())))
        self._agent = agent
        self._player_id = player_id
        self._game = game

    def action_probabilities(self, state, player_id=None):
        if state.is_terminal():
            return {}

        legal_actions = state.legal_actions(self._player_id)
        if not legal_actions:
            return {}

        # Get info state as tensor
        info_state = np.array(
            state.information_state_tensor(self._player_id), dtype=np.float32
        )

        with torch.no_grad():
            obs = torch.tensor(info_state).unsqueeze(0).to(self._agent.device)
            legal_mask = torch.zeros(1, self._agent.num_actions, dtype=torch.bool)
            for a in legal_actions:
                legal_mask[0, a] = True
            legal_mask = legal_mask.to(self._agent.device)

            _, _, _, _, probs = self._agent.get_action_and_value(
                obs, legal_actions_mask=legal_mask
            )
            probs = probs.squeeze(0).cpu().numpy()

        return {a: float(probs[a]) for a in legal_actions}


# ── load agent ───────────────────────────────────────────────────────────────

def load_agent(model_path, game, player_id, device="cpu"):
    info_state_shape = tuple(game.information_state_tensor_shape())
    num_actions = game.num_distinct_actions()

    agent = PPO(
        input_shape=info_state_shape,
        num_actions=num_actions,
        num_players=game.num_players(),
        player_id=player_id,
        num_envs=1,
        steps_per_batch=128,
        device=device,
        agent_fn=PPOAgent,
    )
    agent.network.load_state_dict(torch.load(model_path, map_location=device))
    agent.network.eval()
    print(f"Loaded agent from {model_path}")
    return agent


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    game = pyspiel.load_game("kuhn_poker")

    # Load your trained agents (adjust paths as needed)
    agent0 = load_agent("kuhn_poker_ppo.pth", game, player_id=0)

    # Wrap as OpenSpiel policy
    policy0 = PPOPolicy(game, agent0, player_id=0)

    # For a 2-player game we need both players' policies
    # If you only trained player 0, use a uniform random policy for player 1
    uniform = policy_lib.UniformRandomPolicy(game)

    class MixedPolicy(policy_lib.Policy):
        """Player 0 uses PPO, Player 1 uses uniform random."""
        def __init__(self):
            super().__init__(game, [0, 1])

        def action_probabilities(self, state, player_id=None):
            pid = state.current_player()
            if pid == 0:
                return policy0.action_probabilities(state)
            else:
                return uniform.action_probabilities(state)

    mixed = MixedPolicy()

    # ── compute exploitability ──
    print("\nComputing exploitability...")
    expl = exploitability.exploitability(game, mixed)

    print("\n" + "="*50)
    print(f"  Exploitability: {expl:.6f}")
    print(f"  Nash equilibrium target: 0.000000")
    print(f"  Random policy baseline:  ~0.500000")
    print("="*50)



if __name__ == "__main__":
    main()