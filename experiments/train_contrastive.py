"""
Train Contrastive Encoder for State Representation Learning.

Collects state observations and trains a contrastive encoder
using a SimCLR-style framework with state augmentation.

Usage:
    python experiments/train_contrastive.py
    python experiments/train_contrastive.py --env LunarLander-v2 --latent-dim 32
"""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.env_wrapper import EnvWrapper
from src.representations.contrastive import ContrastiveEncoder, ContrastiveTrainer
from src.utils.config import Config
from src.utils.visualization import Visualizer


def train_contrastive(args):
    """Main contrastive training."""
    config = Config(seed=args.seed)
    device = config.get_device()
    print(f"Using device: {device}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Environment
    env = EnvWrapper(args.env, seed=args.seed, normalize_states=False)
    print(f"Environment: {env}")
    
    # Collect data
    print(f"\nCollecting {args.collect_episodes} episodes...")
    states, actions, rewards, next_states = env.collect_experiences(
        num_episodes=args.collect_episodes,
        policy="random"
    )
    print(f"Collected {len(states)} state observations")
    
    # Build model
    model = ContrastiveEncoder(
        state_dim=env.state_dim,
        latent_dim=args.latent_dim,
        projection_dim=32,
        hidden_dims=[64, 32]
    )
    print(f"\nContrastive Encoder:")
    print(model)
    
    # Train
    trainer = ContrastiveTrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=1e-5,
        temperature=args.temperature,
        noise_std=0.1,
        mask_ratio=0.15,
        device=device
    )
    
    exp_name = f"contrastive_{args.env.replace('-', '_')}"
    
    print(f"\n{'='*60}")
    print(f"Training Contrastive Encoder | Latent Dim: {args.latent_dim}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")
    
    history = trainer.train(
        states=states,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        log_interval=10
    )
    
    # Visualize
    viz = Visualizer(save_dir=args.results_dir)
    
    # Loss curve
    viz.plot_loss_curve(
        history['train_losses'],
        title=f"Contrastive Loss ({args.env})",
        filename=f"{exp_name}_loss.png"
    )
    
    # t-SNE of learned representations
    model.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(states).to(device)
        # Process in batches to avoid memory issues
        batch_size = 1000
        latent_list = []
        for i in range(0, len(state_tensor), batch_size):
            batch = state_tensor[i:i+batch_size]
            latent_batch = model(batch).cpu().numpy()
            latent_list.append(latent_batch)
        latent = np.concatenate(latent_list, axis=0)
    
    viz.plot_tsne(
        latent,
        labels=rewards,
        title=f"Contrastive Latent Space ({args.env}) - Colored by Reward",
        filename=f"{exp_name}_tsne.png"
    )
    
    viz.plot_latent_space_analysis(
        latent, rewards, actions,
        title=f"Contrastive Latent Space Analysis ({args.env})",
        filename=f"{exp_name}_latent_analysis.png"
    )
    
    # Save
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    trainer.save(os.path.join(args.checkpoints_dir, f"{exp_name}.pth"))
    
    print(f"\n{'='*60}")
    print(f"Contrastive training complete!")
    print(f"Final loss: {history['train_losses'][-1]:.6f}")
    print(f"{'='*60}")
    
    env.close()
    return history


def parse_args():
    parser = argparse.ArgumentParser(description='Train Contrastive Encoder')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'LunarLander-v2'])
    parser.add_argument('--latent-dim', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--collect-episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_contrastive(args)
