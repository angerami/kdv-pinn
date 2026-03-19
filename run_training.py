import argparse
import torch
from models import KdV_pinn
from train import train_pinn
from configuration import kdv_config


def main():
    parser = argparse.ArgumentParser(description='Train PINN for 1D KDV equation')
    parser.add_argument('--epochs', type=int, default=None)
    #Add argument for filename used in output filename and any saved plots
    args = parser.parse_args()

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
    torch.manual_seed(training_config.seed)

    model = KdV_pinn(training_config).to(device)
    result = train_pinn(model, training_config, device)
    print("Training complete!")

    torch.save({
        'model_state_dict': result['model'].state_dict(),
        'optimizer_state_dict': result['optimizer'].state_dict(),
        'config': training_config,
        'metrics': result['metrics']
    }, 'pinn_model.pt')
    print("Model saved to pinn_model.pt")
    return result

if __name__ == '__main__':
    main()