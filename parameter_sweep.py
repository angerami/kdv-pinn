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

    # Explicit sweep configurations with interesting collision dynamics
    # Focus on: fast chasers, multiple crossings, cascade collisions, speed variety
    sweep_configs = {
        1: {
            'kappas': [[1.0], [1.5], [2.0], [2.5]],
            'x0s': [[0], [5], [-5], [10]]
        },
        2: {
            'kappas': [
                # Fast chaser scenarios - dramatic catch-ups
                [2.5, 0.8],  # Very fast catching very slow
                [2.0, 1.0],  # Classic fast chaser
                [3.0, 1.5],  # Super fast catching medium
                # Similar speeds - minimal interaction
                [1.5, 1.4],  # Almost matched speeds, should barely interact
                [2.0, 1.9],  # Very close speeds
                # Reverse - slow in front (no collision expected)
                [0.9, 2.0],  # Slow leads, fast behind - shouldn't collide
                # Widely separated
                [3.5, 0.6],  # Extreme speed difference
                [2.2, 1.1],  # 2:1 speed ratio
            ],
            'x0s': [
                [-10, 5],    # Fast far behind, has to chase
                [-8, 0],     # Fast behind slow
                [-15, 3],    # Fast very far behind
                [0, 0.5],    # Nearly overlapping, similar speeds
                [0, 1],      # Slightly separated, similar speeds
                [5, -10],    # Slow in front, fast way behind (no collision)
                [-12, 2],    # Extreme chase scenario
                [-10, 0],    # Fast chaser from behind
            ]
        },
        3: {
            'kappas': [
                # Cascade: fast catches middle, then catches slow
                [3.0, 1.5, 0.8],   # Clear cascade progression
                [2.5, 1.3, 0.7],   # Another cascade
                # One fast, two slow and close
                [3.0, 1.0, 0.9],   # Fast guy catches tight pair
                # Fast guy in middle (by speed, not position)
                [2.5, 0.9, 1.6],   # Slow in middle by speed
                [2.0, 1.0, 1.8],   # Middle is slowest
                # Fast guy way in back catches everyone
                [3.5, 1.2, 1.0],   # One super fast chaser
                # Cluster of similar speeds
                [1.5, 1.4, 1.45],  # Very tight cluster - minimal dynamics
                # Wide spread
                [3.0, 1.5, 0.6],   # Evenly spread speeds
            ],
            'x0s': [
                [-15, -2, 5],      # Fast way back, catches middle then slow
                [-12, -3, 4],      # Cascade scenario
                [-15, 2, 3],       # Fast catches tight pair
                [-10, 3, -4],      # Slow in middle position
                [-8, 5, -6],       # Similar mixed ordering
                [-20, 2, 0],       # Fast very far behind everyone
                [0, 0.5, 1],       # Tight cluster in space
                [-15, -5, 5],      # Evenly spaced
            ]
        },
        5: {
            'kappas': [
                # One super fast guy chasing pack
                [4.0, 1.2, 1.0, 0.8, 0.6],   # Single fast chaser
                [3.5, 1.4, 1.1, 0.9, 0.7],   # Fast chaser with graduated pack
                # Two fast chasers
                [3.0, 2.5, 0.9, 0.8, 0.7],   # Two fast, three slow clustered
                [2.8, 2.2, 1.0, 0.9, 0.8],   # Two fast, tight slow pack
                # Mixed speeds (not monotonic)
                [2.5, 1.0, 2.0, 0.8, 1.5],   # Interleaved speeds
                [3.0, 0.7, 2.0, 1.0, 1.5],   # More speed mixing
            ],
            'x0s': [
                [-25, 3, 5, 7, 9],           # Fast way behind pack
                [-20, 2, 4, 6, 8],           # Fast chaser
                [-15, -12, 4, 6, 8],         # Two fast chasers
                [-12, -10, 3, 5, 7],         # Two fast back, tight pack ahead
                [-10, 5, -8, 8, -2],         # Interleaved positions
                [-15, 10, -5, 3, 0],         # Mixed positions
            ]
        },
        7: {
            'kappas': [
                # Multiple fast chasers
                [4.0, 3.0, 1.0, 0.9, 0.8, 0.7, 0.6],  # Two fast, pack of slow
                [3.5, 2.5, 2.0, 0.9, 0.8, 0.75, 0.7], # Three fast, pack of slow
                # Interleaved
                [3.0, 1.0, 2.5, 0.9, 2.0, 0.8, 1.5],  # Completely mixed speeds
            ],
            'x0s': [
                [-30, -20, 5, 7, 9, 11, 13],          # Two fast way back
                [-25, -15, -10, 6, 8, 10, 12],        # Three fast back
                [-15, 8, -12, 10, -8, 12, -2],        # Interleaved positions
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
                kdv_config.num_epochs = 5000
                kdv_config.num_pretrain_epochs = 2500
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
