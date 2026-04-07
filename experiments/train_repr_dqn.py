"""
Train DQN Agent with Learned State Representations.

First loads a pre-trained encoder (autoencoder or contrastive),
then trains a DQN agent using the learned latent representations
as input instead of raw state observations.

Usage:
    python experiments/train_repr_dqn.py --repr autoencoder
    python experiments/train_repr_dqn.py --repr contrastive --env LunarLander-v2
"""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.env_wrapper import EnvWrapper
from src.agents.dqn_agent import DQNAgent
from src.representations.autoencoder import StateAutoencoder
from src.representations.contrastive import ContrastiveEncoder
from src.utils.config import Config
from src.utils.logger import TrainingLogger
from src.utils.visualization import Visualizer


def evaluate_agent(agent, env, n_episodes=10):
    """Evaluate agent performance."""
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


def load_encoder(repr_type, env_name, latent_dim, checkpoints_dir, device):
    """Load pre-trained encoder."""
    env_suffix = env_name.replace('-', '_')
    
    if repr_type == 'autoencoder':
        checkpoint_path = os.path.join(checkpoints_dir, f"ae_{env_suffix}.pth")
        if not os.path.exists(checkpoint_path):
            # Try VAE
            checkpoint_path = os.path.join(checkpoints_dir, f"vae_{env_suffix}.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No autoencoder checkpoint found. Run train_autoencoder.py first.\n"
                f"Looked for: {checkpoint_path}"
            )
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        variational = checkpoint.get('variational', False)
        saved_latent_dim = checkpoint.get('latent_dim', latent_dim)
        
        # Determine state dim from checkpoint
        first_key = list(checkpoint['model_state_dict'].keys())[0]
        state_dim = checkpoint['model_state_dict']['encoder.encoder.0.weight'].shape[1]
        
        model = StateAutoencoder(
            state_dim=state_dim,
            latent_dim=saved_latent_dim,
            hidden_dims=[64, 32],
            variational=variational
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Return just the encoder part
        encoder = model.encoder
        print(f"Loaded {'VAE' if variational else 'AE'} encoder (latent_dim={saved_latent_dim})")
        return encoder, saved_latent_dim
    
    elif repr_type == 'contrastive':
        checkpoint_path = os.path.join(checkpoints_dir, f"contrastive_{env_suffix}.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No contrastive checkpoint found. Run train_contrastive.py first.\n"
                f"Looked for: {checkpoint_path}"
            )
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_latent_dim = checkpoint.get('latent_dim', latent_dim)
        
        state_dim = checkpoint['model_state_dict']['encoder.encoder.0.weight'].shape[1]
        
        model = ContrastiveEncoder(
            state_dim=state_dim,
            latent_dim=saved_latent_dim,
            projection_dim=32,
            hidden_dims=[64, 32]
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        encoder = model.encoder
        print(f"Loaded Contrastive encoder (latent_dim={saved_latent_dim})")
        return encoder, saved_latent_dim
    
    else:
        raise ValueError(f"Unknown representation type: {repr_type}")


def train_repr_dqn(args):
    """Train DQN with learned representations."""
    config = Config(seed=args.seed)
    device = config.get_device()
    print(f"Using device: {device}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Environment
    env = EnvWrapper(args.env, seed=args.seed, normalize_states=False)
    print(f"Environment: {env}")
    
    # Load pre-trained encoder
    encoder, latent_dim = load_encoder(
        args.repr, args.env, args.latent_dim,
        args.checkpoints_dir, device
    )
    
    # DQN agent with encoder
    agent = DQNAgent(
        state_dim=latent_dim,
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
        encoder=encoder,
    )
    
    exp_name = f"dqn_{args.repr}_{args.env.replace('-', '_')}"
    logger = TrainingLogger(exp_name, log_dir=args.results_dir)
    viz = Visualizer(save_dir=args.results_dir)
    
    print(f"\n{'='*60}")
    print(f"Training DQN with {args.repr} representations on {args.env}")
    print(f"Latent dim: {latent_dim}, Episodes: {args.episodes}")
    print(f"{'='*60}\n")
    
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step_fn()
            
            if loss is not None:
                episode_loss += loss
            
            total_reward += reward
            steps += 1
            state = next_state
        
        agent.end_episode()
        
        avg_loss = episode_loss / max(steps, 1)
        logger.log_episode(total_reward, steps, avg_loss, agent.epsilon)
        
        if episode % args.log_interval == 0:
            logger.print_status(episode, args.episodes)
        
        if episode % args.eval_interval == 0:
            eval_rewards = evaluate_agent(agent, env, n_episodes=10)
            logger.log_evaluation(episode, eval_rewards)
            print(f"  >> Eval: Mean={np.mean(eval_rewards):.2f}, "
                  f"Std={np.std(eval_rewards):.2f}")
    
    # Save results
    logger.save()
    
    viz.plot_training_rewards(
        logger.episode_rewards,
        title=f"DQN + {args.repr.title()} on {args.env}",
        color='#FF9800' if args.repr == 'autoencoder' else '#4CAF50',
        filename=f"{exp_name}_rewards.png"
    )
    
    if logger.losses:
        viz.plot_loss_curve(
            logger.losses,
            title=f"DQN + {args.repr.title()} Loss",
            filename=f"{exp_name}_loss.png"
        )
    
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    agent.save(os.path.join(args.checkpoints_dir, f"{exp_name}.pth"))
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best eval reward: {logger.best_eval_reward:.2f} (Episode {logger.best_episode})")
    print(f"Final avg (last 100): {np.mean(logger.episode_rewards[-100:]):.2f}")
    print(f"{'='*60}")
    
    env.close()
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN with learned representations')
    parser.add_argument('--repr', type=str, required=True,
                        choices=['autoencoder', 'contrastive'],
                        help='Type of representation to use')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'LunarLander-v2'])
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--latent-dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dueling', action='store_true')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_repr_dqn(args)
