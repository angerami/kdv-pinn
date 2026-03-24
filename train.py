"""Training routines for KdV PINN."""
import torch
import matplotlib.pyplot as plt
from sampling import sample_bulk, sample_boundary
from physics import kdv, init_metrics
from scattering import ScatteringData
from plot_utils import _init_interactive_plot, _update_interactive_plot


def extract_means(results):
    """Extract mean values from field results for logging."""
    return {f'mean_{k}': torch.mean(v).detach().cpu().item() for k, v in results.items()}


def calc_residual(model, input, fcn, key):
    """Compute residual between model output and target function."""
    return {f'res_{key}': model(input) - fcn(input)}

def extract_losses_MSE(results, config):
    """Compute weighted MSE losses from PDE residuals.

    Args:
        results: Dictionary from kdv() function
        config: Configuration with loss_types and lambda weights

    Returns:
        L_total: Total weighted loss (for backprop)
        losses: Dictionary of individual loss components (for logging)
    """
    loss_types = config.loss_types
    losses = {}
    L_total = 0

    for k in loss_types:
        L_k = torch.mean(results[f'res_{k}']**2)
        lambda_k = getattr(config, f'lambda_{k}')
        L_total += lambda_k * L_k
        losses[f'L_{k}'] = lambda_k * L_k.detach().cpu().item()

    if 'res_S' in results:
        L_S = torch.mean(results['res_S']**2)
        losses['L_S'] = L_S.detach().cpu().item()

    return L_total, losses


def pretrain(model, fcn, config, device, num_epochs=200):
    """Pretrain model to match analytic solution (supervised learning).

    Args:
        model: Neural network to train
        fcn: Target function (e.g., from ScatteringData)
        config: Configuration object
        device: PyTorch device
        num_epochs: Number of pretraining epochs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    grid = sample_bulk(config, resample=False).to(device)
    grid.requires_grad_(True)
    target = fcn(grid).detach()
    plot_interval = getattr(config, 'plot_interval', 50)

    for epoch in range(num_epochs):
        model.train()
        u = model(grid)
        loss = ((u - target)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % plot_interval == 0:
            print(f'Pretrain Epoch {epoch:4d} | L_total: {loss.item():.3e}')


def train_pinn(model, config, device, start_epoch=0, optimizer_states=None, interactive=False, save_plot=None):
    """Train PINN using physics-informed loss.

    Args:
        model: Neural network to train
        config: Configuration object with training parameters
        device: PyTorch device
        start_epoch: Starting epoch (for resuming training)
        optimizer_states: Optional optimizer state dict (for resuming)
        interactive: If True, show interactive plots in Jupyter
        save_plot: If provided, save training plot to this path

    Returns:
        Dictionary with trained model, metrics, and optimizer
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.eta_min
    )
    if optimizer_states is not None:
        optimizer.load_state_dict(optimizer_states)

    plot_interval = getattr(config, 'plot_interval', 50)

    # Initialize interactive plotting (for Jupyter)
    clear_output, display = None, None
    if interactive:
        clear_output, display = _init_interactive_plot()
        if clear_output is None:
            print("Warning: Interactive plotting requires IPython/Jupyter. Disabling interactive mode.")
            interactive = False

    # Setup evaluation grid and boundary conditions
    input_eval = sample_bulk(config, resample=False, num_samp=config.num_samp_eval).to(device)
    sd = ScatteringData(config.kappas, config.x0s, R=None, use_tau=True)
    fcn = sd.forward_fcn

    # Initialize metrics tracking
    metrics = init_metrics()
    metrics.update({'L_total': [], 'L_KDV': [], 'L_IC': [], 'L_BC': [], 'L_S': []})

    # input_BC = sample_boundary(config, resample=False).to(device)
    # input_bulk = sample_bulk(config=config, resample=True).to(device)
    #TRAINING LOOP
    print(f"Training for {config.num_epochs} epochs (from epoch {start_epoch})")
    for epoch in range(start_epoch, config.num_epochs):
        model.train()

        # Sample different domains for different loss terms
        input_bulk = sample_bulk(config=config, resample=True).to(device)
        input_bulk.requires_grad_(True)

        u = model(input_bulk)
        results = kdv(u, input_bulk)

        # update results for IC and boundary conditions
        input_IC = sample_boundary(config, resample=True, ic_only=True).to(device)
        input_IC.requires_grad_(True)
        results.update(calc_residual(model, input_IC, fcn=fcn, key='IC'))

        input_BC = sample_boundary(config, resample=True, bc_only=True).to(device)
        input_BC.requires_grad_(True)
        results.update(calc_residual(model, input_BC, fcn=fcn, key='BC'))

        results.update({'res_S' : u - fcn(input_bulk)})

        loss, loss_dict = extract_losses_MSE(results, config)
        loss_dict.update(extract_means(results))

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track metrics
        for k in metrics.keys():
            if k == 'L_total':
                metrics[k].append(loss.detach().cpu().item())
            elif k in loss_dict:
                metrics[k].append(loss_dict[k])

        #More plotting and loggging
        if epoch % plot_interval == 0:
            print_keys = ['L_total', 'L_KDV', 'L_IC', 'L_BC', 'L_S', 'mean_u', 'mean_u_t', 'mean_u_x']
            print_items = [(k, loss_dict[k]) for k in print_keys if k in loss_dict]
            print(f'Epoch {epoch:4d} | ' + ' | '.join([f'{k} : {v:.3e}' for k, v in print_items]))
            if interactive:
                _update_interactive_plot(model, input_eval, config, metrics)
            if save_plot:
                from plot_utils import plot_results
                model.eval()
                with torch.no_grad():
                    fig = plot_results(model, input_eval, config, metrics, filename=save_plot)
                    plt.close(fig)
                model.train()
                

    return {'model': model, 'metrics': metrics, 'optimizer': optimizer}


def main():
    """Main training script (CLI entry point)."""
    import argparse
    from models import KdV_pinn
    from configuration import kdv_config, config_to_dict

    parser = argparse.ArgumentParser(description='Train PINN for 1D KDV equation')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    args = parser.parse_args()

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device.")

    # Setup configuration
    training_config = kdv_config
    if args.epochs is not None:
        training_config.num_epochs = args.epochs
    torch.manual_seed(training_config.seed)

    # Train model
    model = KdV_pinn(training_config).to(device)
    result = train_pinn(model, training_config, device)
    print("Training complete!")

    # Save model
    torch.save({
        'model_state_dict': result['model'].state_dict(),
        'optimizer_state_dict': result['optimizer'].state_dict(),
        'config': config_to_dict(training_config),
        'metrics': result['metrics']
    }, 'pinn_model.pt')
    print("Model saved to pinn_model.pt")

    return result


if __name__ == '__main__':
    main()