"""
Train DQN Agent with Raw State Observations.

This script trains a standard DQN agent on CartPole-v1 and LunarLander-v2
using raw state observations as input to the Q-network.

Usage:
    python experiments/train_dqn.py
    python experiments/train_dqn.py --env LunarLander-v2 --episodes 1000
"""

import sys
import os
import argparse
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.env_wrapper import EnvWrapper
from src.agents.dqn_agent import DQNAgent
from src.utils.config import Config
from src.utils.logger import TrainingLogger
from src.utils.visualization import Visualizer


def evaluate_agent(agent: DQNAgent, env: EnvWrapper, n_episodes: int = 10):
    """Evaluate agent performance without exploration."""
    rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
    return rewards


def train_dqn(args):
    """Main DQN training loop."""
    # Configuration
    config = Config(seed=args.seed)
    device = config.get_device()
    print(f"Using device: {device}")
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Environment
    env = EnvWrapper(args.env, seed=args.seed, normalize_states=False)
    print(f"Environment: {env}")
    
    # Agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dims=[128, 128],
        lr=args.lr,
        gamma=0.99,
        tau=0.005,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        use_dueling=args.dueling,
        device=device,
    )
    
    # Logger and Visualizer
    exp_name = f"dqn_raw_{args.env.replace('-', '_')}"
    if args.dueling:
        exp_name = f"dueling_{exp_name}"
    
    logger = TrainingLogger(exp_name, log_dir=args.results_dir)
    viz = Visualizer(save_dir=args.results_dir)
    
    print(f"\n{'='*60}")
    print(f"Training {'Dueling ' if args.dueling else ''}DQN on {args.env}")
    print(f"Episodes: {args.episodes}, LR: {args.lr}")
    print(f"{'='*60}\n")
    
    # Training loop
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        steps = 0
        done = False
        
        while not done:
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step_fn()
            
            if loss is not None:
                episode_loss += loss
            
            total_reward += reward
            steps += 1
            state = next_state
        
        agent.end_episode()
        
        # Log
        avg_loss = episode_loss / max(steps, 1)
        logger.log_episode(total_reward, steps, avg_loss, agent.epsilon)
        
        # Print status
        if episode % args.log_interval == 0:
            logger.print_status(episode, args.episodes)
        
        # Evaluation
        if episode % args.eval_interval == 0:
            eval_rewards = evaluate_agent(agent, env, n_episodes=10)
            logger.log_evaluation(episode, eval_rewards)
            print(f"  >> Eval: Mean={np.mean(eval_rewards):.2f}, "
                  f"Std={np.std(eval_rewards):.2f}, "
                  f"Best={logger.best_eval_reward:.2f}")
    
    # Save results
    logger.save()
    
    # Generate plots
    viz.plot_training_rewards(
        logger.episode_rewards,
        title=f"{'Dueling ' if args.dueling else ''}DQN on {args.env} (Raw States)",
        filename=f"{exp_name}_rewards.png"
    )
    
    if logger.losses:
        viz.plot_loss_curve(
            logger.losses,
            title=f"{'Dueling ' if args.dueling else ''}DQN Training Loss",
            filename=f"{exp_name}_loss.png"
        )
    
    if logger.epsilons:
        viz.plot_epsilon_decay(
            logger.epsilons,
            filename=f"{exp_name}_epsilon.png"
        )
    
    # Save agent
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    agent.save(os.path.join(args.checkpoints_dir, f"{exp_name}.pth"))
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best eval reward: {logger.best_eval_reward:.2f} (Episode {logger.best_episode})")
    print(f"Final avg reward (last 100): {np.mean(logger.episode_rewards[-100:]):.2f}")
    print(f"{'='*60}")
    
    env.close()
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN with raw states')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'LunarLander-v2'])
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dueling', action='store_true',
                        help='Use Dueling DQN architecture')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_dqn(args)
