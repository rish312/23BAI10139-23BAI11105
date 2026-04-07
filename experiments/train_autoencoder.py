"""
Train Autoencoder for State Representation Learning.

Collects state observations from the environment and trains
an autoencoder (standard AE or VAE) to learn compact representations.

Usage:
    python experiments/train_autoencoder.py
    python experiments/train_autoencoder.py --env LunarLander-v2 --vae
"""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.env_wrapper import EnvWrapper
from src.representations.autoencoder import StateAutoencoder, AutoencoderTrainer
from src.utils.config import Config
from src.utils.visualization import Visualizer


def train_autoencoder(args):
    """Main autoencoder training."""
    config = Config(seed=args.seed)
    device = config.get_device()
    print(f"Using device: {device}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Environment for data collection
    env = EnvWrapper(args.env, seed=args.seed, normalize_states=False)
    print(f"Environment: {env}")
    
    # Collect experience data
    print(f"\nCollecting {args.collect_episodes} episodes of experience data...")
    states, actions, rewards, next_states = env.collect_experiences(
        num_episodes=args.collect_episodes,
        policy="random"
    )
    print(f"Collected {len(states)} state observations")
    
    # Build autoencoder
    ae_type = "VAE" if args.vae else "AE"
    model = StateAutoencoder(
        state_dim=env.state_dim,
        latent_dim=args.latent_dim,
        hidden_dims=[64, 32],
        dropout=0.0,
        variational=args.vae
    )
    print(f"\n{ae_type} Architecture:")
    print(model)
    
    # Train
    trainer = AutoencoderTrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=1e-5,
        kl_weight=0.001,
        device=device
    )
    
    exp_name = f"{'vae' if args.vae else 'ae'}_{args.env.replace('-', '_')}"
    
    print(f"\n{'='*60}")
    print(f"Training {ae_type} | Latent Dim: {args.latent_dim}")
    print(f"{'='*60}\n")
    
    history = trainer.train(
        states=states,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        log_interval=10
    )
    
    # Visualize
    viz = Visualizer(save_dir=args.results_dir)
    
    # Loss curves
    viz.plot_autoencoder_losses(
        history['train_losses'],
        history['val_losses'],
        title=f"{ae_type} Training on {args.env}",
        filename=f"{exp_name}_losses.png"
    )
    
    # Reconstruction quality
    model.eval()
    with torch.no_grad():
        sample_states = torch.FloatTensor(states[:100]).to(device)
        if args.vae:
            reconstructed, _, _, _ = model(sample_states)
        else:
            reconstructed, _ = model(sample_states)
        reconstructed = reconstructed.cpu().numpy()
    
    feature_names = None
    if args.env == 'CartPole-v1':
        feature_names = ['Cart Pos', 'Cart Vel', 'Pole Angle', 'Pole Vel']
    elif args.env == 'LunarLander-v2':
        feature_names = ['X', 'Y', 'Vx', 'Vy', 'Angle', 'AngVel', 'Left', 'Right']
    
    viz.plot_reconstruction(
        states[:100], reconstructed,
        feature_names=feature_names,
        n_samples=5,
        title=f"{ae_type} Reconstruction ({args.env})",
        filename=f"{exp_name}_reconstruction.png"
    )
    
    # t-SNE of learned representations
    model.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(states).to(device)
        latent = model.encode(state_tensor).cpu().numpy()
    
    viz.plot_tsne(
        latent,
        labels=rewards,
        title=f"{ae_type} Latent Space ({args.env}) - Colored by Reward",
        filename=f"{exp_name}_tsne.png"
    )
    
    # Latent space analysis
    viz.plot_latent_space_analysis(
        latent, rewards, actions,
        title=f"{ae_type} Latent Space Analysis ({args.env})",
        filename=f"{exp_name}_latent_analysis.png"
    )
    
    # Save model
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    trainer.save(os.path.join(args.checkpoints_dir, f"{exp_name}.pth"))
    
    print(f"\n{'='*60}")
    print(f"{ae_type} training complete!")
    print(f"Final train loss: {history['train_losses'][-1]:.6f}")
    print(f"Final val loss: {history['val_losses'][-1]:.6f}")
    print(f"{'='*60}")
    
    env.close()
    return history


def parse_args():
    parser = argparse.ArgumentParser(description='Train Autoencoder for state representation')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'LunarLander-v2'])
    parser.add_argument('--latent-dim', type=int, default=16)
    parser.add_argument('--vae', action='store_true', help='Use Variational Autoencoder')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--collect-episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_autoencoder(args)
