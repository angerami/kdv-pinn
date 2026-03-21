import torch
import matplotlib.pyplot as plt
from sampling import sample_bulk, sample_boundary
from physics import kdv, init_metrics, soliton_from_config
from plot_utils import _init_interactive_plot, _update_interactive_plot

def extract_means(results):
    return {f'mean_{k}' : torch.mean(v).detach().cpu().item() for k,v in results.items()}

def calc_residual(model, input, fcn, key):
    return {f'res_{key}' : model(input) - fcn(input)}

def extract_losses_MSE(results, config):
    loss_types = config.loss_types
    losses = {}
    L_total = 0
    for k in loss_types:
        L_k = torch.mean(results[f'res_{k}']**2)
        lambda_k = getattr(config, f'lambda_{k}') 
        L_total += lambda_k * L_k
        losses[f'L_{k}'] = lambda_k * L_k.detach().cpu().item()
    return L_total, losses



def pretrain(model, fcn, config, device, num_epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    grid = sample_bulk(config, resample=False).to(device)
    grid.requires_grad_(True)
    target = fcn(grid).detach()
    for epoch in range(num_epochs):
        model.train()
        u = model(grid)
        loss = ((u - target)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_pinn(model, config, device, start_epoch=0, optimizer_states=None, interactive=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.eta_min
    )
    if optimizer_states is not None:
        optimizer.load_state_dict(optimizer_states)

    #Plot initialization
    plot_interval = getattr(config, 'plot_interval', 50)

    clear_output, display = None, None
    if interactive:
        clear_output, display = _init_interactive_plot()
        if clear_output is None:
            print("Warning: Interactive plotting requires IPython/Jupyter environment. Disabling interactive mode.")
            interactive = False

    input_eval = sample_bulk(config, resample=False, num_samp=config.num_samp_eval).to(device)
    fcn = soliton_from_config(config)
    # Initialize metrics tracking
    metrics = init_metrics()
    metrics.update({'L_total': [], 'L_kdv': [], 'L_BC': []})

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
        # these are generic residuals wrt fcn evaluated on the boundary
        input_BC = sample_boundary(config, resample=True).to(device)
        input_BC.requires_grad_(True) #NEEDED?
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
            print(f'Epoch {epoch:4d} | ' + ' | '.join([f'{k} : {v:.3e}' for k, v in loss_dict.items()]))
            if interactive:
                _update_interactive_plot(model, input_eval, config, metrics)
                

    return {'model': model, 'metrics': metrics, 'optimizer': optimizer}