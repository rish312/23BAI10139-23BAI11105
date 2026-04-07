"""
Compare Results Across All Methods.

Loads training logs from all experiments and generates
comparison plots and summary statistics.

Usage:
    python experiments/compare_results.py
    python experiments/compare_results.py --env CartPole-v1
"""

import sys
import os
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import TrainingLogger
from src.utils.visualization import Visualizer


def load_results(results_dir: str, env_name: str) -> dict:
    """Load all experiment results for an environment."""
    env_suffix = env_name.replace('-', '_')
    results = {}
    
    # Expected log files
    experiment_map = {
        f'dqn_raw_{env_suffix}': 'DQN (Raw States)',
        f'dqn_autoencoder_{env_suffix}': 'DQN + Autoencoder',
        f'dqn_contrastive_{env_suffix}': 'DQN + Contrastive',
        f'dueling_dqn_raw_{env_suffix}': 'Dueling DQN (Raw)',
    }
    
    for filename_base, label in experiment_map.items():
        filepath = os.path.join(results_dir, f"{filename_base}_log.json")
        if os.path.exists(filepath):
            logger = TrainingLogger.load(filepath)
            results[label] = logger
            print(f"  Loaded: {label} ({len(logger.episode_rewards)} episodes)")
        else:
            print(f"  Not found: {label} ({filepath})")
    
    return results


def print_summary_table(results: dict):
    """Print comparison summary table."""
    print(f"\n{'='*80}")
    print(f"{'Method':<30} {'Episodes':>8} {'Best Eval':>10} {'Final Avg':>10} {'Max':>8}")
    print(f"{'='*80}")
    
    for method, logger in results.items():
        n_episodes = len(logger.episode_rewards)
        best_eval = logger.best_eval_reward
        final_avg = np.mean(logger.episode_rewards[-100:]) if n_episodes >= 100 else np.mean(logger.episode_rewards)
        max_reward = max(logger.episode_rewards)
        
        print(f"{method:<30} {n_episodes:>8} {best_eval:>10.2f} {final_avg:>10.2f} {max_reward:>8.2f}")
    
    print(f"{'='*80}")


def compare_results(args):
    """Generate comparison visualizations."""
    viz = Visualizer(save_dir=args.results_dir)
    env_suffix = args.env.replace('-', '_')
    
    print(f"\nLoading results for {args.env}...")
    results = load_results(args.results_dir, args.env)
    
    if len(results) == 0:
        print("\nNo results found! Run training experiments first:")
        print("  python experiments/train_dqn.py")
        print("  python experiments/train_autoencoder.py")
        print("  python experiments/train_contrastive.py")
        print("  python experiments/train_repr_dqn.py --repr autoencoder")
        print("  python experiments/train_repr_dqn.py --repr contrastive")
        return
    
    # Print summary
    print_summary_table(results)
    
    if len(results) < 2:
        print("\nNeed at least 2 experiment results for comparison.")
        print("Run more experiments and try again.")
        return
    
    # Comparison reward curves
    reward_dict = {name: logger.episode_rewards for name, logger in results.items()}
    viz.plot_comparison(
        reward_dict,
        title=f"Method Comparison: {args.env}",
        window=50,
        filename=f"comparison_rewards_{env_suffix}.png"
    )
    
    # Bar chart of final performance
    final_rewards = {}
    for method, logger in results.items():
        rewards = logger.episode_rewards
        last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
        final_rewards[method] = (np.mean(last_100), np.std(last_100))
    
    viz.plot_bar_comparison(
        final_rewards,
        title=f"Final Performance: {args.env}",
        filename=f"comparison_bar_{env_suffix}.png"
    )
    
    # Save comparison summary as JSON
    summary = {}
    for method, logger in results.items():
        rewards = logger.episode_rewards
        last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
        summary[method] = {
            'total_episodes': len(rewards),
            'best_eval_reward': logger.best_eval_reward,
            'best_episode': logger.best_episode,
            'final_mean_reward': float(np.mean(last_100)),
            'final_std_reward': float(np.std(last_100)),
            'max_reward': float(max(rewards)),
            'convergence_episode': _find_convergence(rewards, threshold=0.9),
        }
    
    summary_path = os.path.join(args.results_dir, f"comparison_summary_{env_suffix}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    print(f"\nComparison complete! Check {args.results_dir}/ for plots.")


def _find_convergence(rewards: list, threshold: float = 0.9, window: int = 50) -> int:
    """Find the episode where the agent first reaches threshold of max performance."""
    if len(rewards) < window:
        return len(rewards)
    
    max_reward = max(rewards)
    target = threshold * max_reward
    
    for i in range(window, len(rewards)):
        avg = np.mean(rewards[i - window:i])
        if avg >= target:
            return i
    
    return len(rewards)


def parse_args():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'LunarLander-v2'])
    parser.add_argument('--results-dir', type=str, default='results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    compare_results(args)
