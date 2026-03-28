"""Main entry point for KdV PINN training and validation."""
import argparse
import os
import torch
from types import SimpleNamespace
from models import KdV_pinn
from train import train_pinn, pretrain
from scattering import ScatteringData, SchrodingerSolver
from configuration import kdv_config, config_to_dict, load_config
from validation import (
    validate_run,
    validate_from_checkpoint,
    validate_analytical_only
)


def main():
    parser = argparse.ArgumentParser(
        description='Train and validate PINN for KdV equation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full validation with animation
  python run_training.py --mode validate --config config.json --animate

  # Generate outputs from existing model (no re-training)
  python run_training.py --mode validate --load-model output/pinn_model.pt

  # Run analytical-only mode (no training)
  python run_training.py --mode analytical --config config.json --animate

  # Simple training run
  python run_training.py --mode train --epochs 5000
        """
    )

    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'validate', 'analytical'],
                       help='Operation mode: train (simple training), validate (full validation), analytical (analytical solution only)')

    # Configuration
    parser.add_argument('--config', type=str, nargs='+',
                       help='Path(s) to JSON configuration file(s). Can specify multiple files.')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for plots and results (overrides config file)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--pretrain-epochs', type=int, default=None,
                       help='Number of pretraining epochs (overrides config)')

    # Model loading
    parser.add_argument('--load-model', type=str,
                       help='Path to saved model checkpoint (.pt file) to load and generate outputs without re-training')

    # Animation
    parser.add_argument('--animate', action='store_true',
                       help='Generate animation of potential and eigenfunctions')

    args = parser.parse_args()

    # Handle validation mode with model loading (no training)
    if args.load_model:
        print(f"Loading model from {args.load_model} for validation...")
        validate_from_checkpoint(
            args.load_model,
            output_dir=args.output_dir,
            generate_anim=args.animate
        )
        return

    # Handle analytical-only mode
    if args.mode == 'analytical':
        # Process config files
        if args.config:
            config_files = args.config
            print(f"Running analytical validation on {len(config_files)} configuration(s)")

            for i, config_path in enumerate(config_files, 1):
                print(f"\n{'='*80}")
                print(f"Configuration {i}/{len(config_files)}: {config_path}")
                print(f"{'='*80}")

                config = load_config(config_path)

                # Override epochs to 0 for analytical mode
                config.num_epochs = 0
                config.num_pretrain_epochs = 0

                # Determine output directory
                if args.output_dir:
                    if len(config_files) == 1:
                        output_dir = args.output_dir
                    else:
                        config_name = getattr(config, 'name', f'run_{i}')
                        output_dir = os.path.join(args.output_dir, config_name)
                else:
                    output_dir = None

                try:
                    validate_analytical_only(
                        config=config,
                        output_dir=output_dir,
                        generate_anim=args.animate
                    )
                    print(f"\n✓ Configuration {i}/{len(config_files)} completed successfully")
                except Exception as e:
                    print(f"\n✗ Configuration {i}/{len(config_files)} failed with error:")
                    print(f"  {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"\n{'='*80}")
            print(f"All configurations complete: {len(config_files)} total")
            print(f"{'='*80}")
        else:
            # Use default config
            validate_analytical_only(
                config=None,
                output_dir=args.output_dir,
                generate_anim=args.animate
            )
        return

    # Handle full validation mode
    if args.mode == 'validate':
        if args.config:
            config_files = args.config
            print(f"Running validation on {len(config_files)} configuration(s)")

            for i, config_path in enumerate(config_files, 1):
                print(f"\n{'='*80}")
                print(f"Configuration {i}/{len(config_files)}: {config_path}")
                print(f"{'='*80}")

                config = load_config(config_path)

                # Override epochs if specified
                if args.epochs is not None:
                    config.num_epochs = args.epochs
                if args.pretrain_epochs is not None:
                    config.num_pretrain_epochs = args.pretrain_epochs

                # Determine output directory
                if args.output_dir:
                    if len(config_files) == 1:
                        output_dir = args.output_dir
                    else:
                        config_name = getattr(config, 'name', f'run_{i}')
                        output_dir = os.path.join(args.output_dir, config_name)
                else:
                    output_dir = None

                try:
                    validate_run(
                        config=config,
                        output_dir=output_dir,
                        generate_anim=args.animate
                    )
                    print(f"\n✓ Configuration {i}/{len(config_files)} completed successfully")
                except Exception as e:
                    print(f"\n✗ Configuration {i}/{len(config_files)} failed with error:")
                    print(f"  {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"\n{'='*80}")
            print(f"All configurations complete: {len(config_files)} total")
            print(f"{'='*80}")
        else:
            # Use default config
            config = SimpleNamespace(**vars(kdv_config))
            if args.epochs is not None:
                config.num_epochs = args.epochs
            if args.pretrain_epochs is not None:
                config.num_pretrain_epochs = args.pretrain_epochs

            validate_run(
                config=config,
                output_dir=args.output_dir,
                generate_anim=args.animate
            )
        return

    # Handle simple training mode (legacy behavior)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device.")

    training_config = kdv_config
    if args.epochs is not None:
        training_config.num_epochs = args.epochs
    if args.pretrain_epochs is not None:
        training_config.num_pretrain_epochs = args.pretrain_epochs

    torch.manual_seed(training_config.seed)

    model = KdV_pinn(training_config).to(device)
    result = train_pinn(model, training_config, device)
    print("Training complete!")

    output_path = args.output_dir if args.output_dir else 'pinn_model.pt'
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, 'pinn_model.pt')

    torch.save({
        'model_state_dict': result['model'].state_dict(),
        'optimizer_state_dict': result['optimizer'].state_dict(),
        'config': config_to_dict(training_config),
        'metrics': result['metrics']
    }, output_path)
    print(f"Model saved to {output_path}")
    return result


if __name__ == '__main__':
    main()