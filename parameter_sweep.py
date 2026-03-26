"""Parameter sweep script for KdV PINN validation runs.

This script performs overnight parameter sweeps over different numbers of solitons,
kappas, and initial positions to generate a comprehensive dataset of validation runs.
"""
import os
import numpy as np
import itertools
from validation import validate_run

def run_parameter_sweep(base_dir='sweep_results'):
    """Run parameter sweep over multiple soliton configurations.

    Sweeps over:
    - Number of solitons: n = 1, 2, 3, 5, 7
    - Different kappa values for each n
    - Different x0 positions for each n

    Args:
        base_dir: Base directory for all sweep results
    """
    os.makedirs(base_dir, exist_ok=True)

    # Define parameter ranges for each number of solitons
    sweep_configs = {
        1: {
            'kappas': [[1.0], [1.2], [1.5], [1.8], [2.0]],
            'x0s': [[0], [5], [-5], [8], [-8]]
        },
        2: {
            'kappas': [
                [1.8, 1.3],  # Standard: large to small, ordered positions
                [2.0, 1.0],
                [1.5, 1.0],
                [1.8, 1.0],
                [2.0, 1.5],
                [1.8, 1.3],  # Shuffled positions
                [2.0, 1.0],  # Mixed ordering
            ],
            'x0s': [
                [8, 7],      # Standard
                [10, 5],
                [8, 0],
                [10, 0],
                [5, 0],
                [5, 10],     # Reversed: slower in front
                [0, 8],      # Mixed
            ]
        },
        3: {
            'kappas': [
                [1.8, 1.3, 0.8],  # Standard: large to small
                [2.0, 1.5, 1.0],
                [1.5, 1.2, 0.9],
                [2.0, 1.2, 0.8],
                [1.8, 1.5, 1.0],
                [1.8, 1.3, 0.8],  # Shuffled positions
                [2.0, 1.5, 1.0],  # Reversed positions
                [1.5, 1.2, 0.9],  # Random order
            ],
            'x0s': [
                [10, 5, 0],       # Standard: ordered
                [12, 6, 0],
                [8, 4, 0],
                [10, 5, -5],
                [12, 8, 4],
                [5, 0, 10],       # Shuffled: middle, slow, fast
                [0, 5, 10],       # Reversed: slowest in front
                [10, 0, 5],       # Random: fast, slow, middle
            ]
        },
        5: {
            'kappas': [
                [2.0, 1.8, 1.5, 1.2, 0.9],  # Standard: large to small
                [2.0, 1.6, 1.3, 1.0, 0.7],
                [1.8, 1.5, 1.2, 0.9, 0.6],
                [2.2, 1.8, 1.4, 1.0, 0.8],
                [2.0, 1.8, 1.5, 1.2, 0.9],  # Shuffled positions
                [2.0, 1.6, 1.3, 1.0, 0.7],  # Reversed positions
            ],
            'x0s': [
                [15, 12, 8, 4, 0],          # Standard: ordered
                [18, 14, 10, 6, 2],
                [16, 12, 8, 4, -2],
                [20, 15, 10, 5, 0],
                [8, 0, 15, 4, 12],          # Shuffled: middle, slow, fastest, slow-mid, fast-mid
                [0, 4, 8, 12, 15],          # Reversed: slowest in front
            ]
        },
        7: {
            'kappas': [
                [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8],  # Standard: large to small
                [2.2, 1.9, 1.6, 1.3, 1.0, 0.8, 0.6],
                [2.0, 1.7, 1.5, 1.3, 1.1, 0.9, 0.7],
                [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8],  # Shuffled positions
                [2.2, 1.9, 1.6, 1.3, 1.0, 0.8, 0.6],  # Reversed positions
            ],
            'x0s': [
                [20, 16, 12, 8, 4, 0, -4],             # Standard: ordered
                [24, 20, 16, 12, 8, 4, 0],
                [22, 18, 14, 10, 6, 2, -2],
                [12, -4, 20, 4, 16, 8, 0],             # Shuffled: middle, slowest, fastest, etc
                [-4, 0, 4, 8, 12, 16, 20],             # Reversed: slowest in front
            ]
        }
    }

    # Track all runs
    total_runs = sum(len(cfg['kappas']) for cfg in sweep_configs.values())
    run_count = 0
    successful_runs = []
    failed_runs = []

    print(f"Starting parameter sweep with {total_runs} total configurations")
    print("=" * 80)

    # Run sweep for each soliton number
    for n_solitons in [1, 2, 3, 5, 7]:
        config = sweep_configs[n_solitons]
        kappa_list = config['kappas']
        x0_list = config['x0s']

        # Run all combinations for this soliton number
        for idx, (kappas, x0s) in enumerate(zip(kappa_list, x0_list)):
            run_count += 1

            # Create descriptive name for this run
            kappa_str = '_'.join([f'{k:.2f}' for k in kappas])
            x0_str = '_'.join([f'{x:.1f}' for x in x0s])
            run_name = f'n{n_solitons}_run{idx:02d}_k{kappa_str}_x{x0_str}'
            output_dir = os.path.join(base_dir, run_name)

            print(f"\n[{run_count}/{total_runs}] Running: {run_name}")
            print(f"  Kappas: {kappas}")
            print(f"  x0s: {x0s}")

            try:
                # Import here to set config per run
                from configuration import kdv_config

                # Configure this run
                kdv_config.kappas = kappas
                kdv_config.x0s = x0s
                kdv_config.num_epochs = 5
                kdv_config.num_pretrain_epochs = 5
                kdv_config.num_samp_bulk = 96
                kdv_config.num_samp_eval = 128
                kdv_config.plot_interval = 50
                kdv_config.MLP = [2, 128, 128, 128, 1]
                kdv_config.lr = 1e-3
                kdv_config.T = 1
                kdv_config.L = 10
                kdv_config.Lmax = 15
                kdv_config.lambda_BC = 1
                kdv_config.lambda_kdv = 1
                kdv_config.vmin = 0
                kdv_config.vmax = 3

                # Mark config as pre-configured so validate_run doesn't override
                kdv_config._configured = True

                # Run validation
                metrics = validate_run(output_dir=output_dir)

                # Clear the flag for next run
                delattr(kdv_config, '_configured')

                successful_runs.append({
                    'name': run_name,
                    'n_solitons': n_solitons,
                    'kappas': kappas,
                    'x0s': x0s,
                    'metrics': metrics
                })
                print(f"  ✓ SUCCESS")

            except Exception as e:
                print(f"  ✗ FAILED: {str(e)}")
                failed_runs.append({
                    'name': run_name,
                    'n_solitons': n_solitons,
                    'kappas': kappas,
                    'x0s': x0s,
                    'error': str(e)
                })

    # Write summary
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"Successful runs: {len(successful_runs)}/{total_runs}")
    print(f"Failed runs: {len(failed_runs)}/{total_runs}")

    summary_file = os.path.join(base_dir, 'sweep_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("PARAMETER SWEEP SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total runs: {total_runs}\n")
        f.write(f"Successful: {len(successful_runs)}\n")
        f.write(f"Failed: {len(failed_runs)}\n\n")

        f.write("SUCCESSFUL RUNS:\n")
        f.write("-" * 80 + "\n")
        for run in successful_runs:
            f.write(f"\n{run['name']}\n")
            f.write(f"  N solitons: {run['n_solitons']}\n")
            f.write(f"  Kappas: {run['kappas']}\n")
            f.write(f"  x0s: {run['x0s']}\n")

            # Write key metrics
            if run['metrics']:
                for key in ['u_mae', 'u_rmse']:
                    if key in run['metrics']:
                        f.write(f"  {key}: {run['metrics'][key]:.6e}\n")

        if failed_runs:
            f.write("\n\nFAILED RUNS:\n")
            f.write("-" * 80 + "\n")
            for run in failed_runs:
                f.write(f"\n{run['name']}\n")
                f.write(f"  N solitons: {run['n_solitons']}\n")
                f.write(f"  Kappas: {run['kappas']}\n")
                f.write(f"  x0s: {run['x0s']}\n")
                f.write(f"  Error: {run['error']}\n")

    print(f"\nSummary written to: {summary_file}")
    print(f"All results saved to: {base_dir}/")

    return successful_runs, failed_runs


if __name__ == "__main__":
    import torch

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("KdV PINN Parameter Sweep")
    print("This will run overnight. Results will be saved to sweep_results/")
    print("\nStarting sweep...\n")

    successful, failed = run_parameter_sweep()

    print("\nSweep complete! Check sweep_results/ for all plots and metrics.")
