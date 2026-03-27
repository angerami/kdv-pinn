"""Interactive browser-based viewer for parameter sweep results.

Allows real-time monitoring of validation runs and comparison across different
hyperparameter configurations.
"""
import os
import time
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, request
import json

app = Flask(__name__)

# Global configuration
SWEEP_DIR = 'sweep_results'
REFRESH_INTERVAL = 2000  # milliseconds

def parse_run_name(run_name):
    """Extract parameters from run directory name.

    Format: n{N}_run{idx}_k{kappas}_x{x0s}
    Example: n2_run00_k1.80_1.30_x8.0_7.0
    """
    try:
        parts = run_name.split('_')
        n_solitons = int(parts[0][1:])  # Extract N from 'nN'
        run_idx = int(parts[1][3:])  # Extract idx from 'runXX'

        # Extract kappas
        k_idx = next(i for i, p in enumerate(parts) if p.startswith('k'))
        x_idx = next(i for i, p in enumerate(parts) if p.startswith('x'))

        kappa_parts = parts[k_idx:x_idx]
        kappas = [float(k[1:]) if i == 0 else float(k) for i, k in enumerate(kappa_parts)]

        x0_parts = parts[x_idx:]
        x0s = [float(x[1:]) if i == 0 else float(x) for i, x in enumerate(x0_parts)]

        return {
            'n_solitons': n_solitons,
            'run_idx': run_idx,
            'kappas': kappas,
            'x0s': x0s
        }
    except:
        return None

def get_available_runs(sweep_dir=SWEEP_DIR):
    """Get list of all validation runs with their parameters."""
    if not os.path.exists(sweep_dir):
        return []

    runs = []
    for item in os.listdir(sweep_dir):
        item_path = os.path.join(sweep_dir, item)
        if os.path.isdir(item_path) and item.startswith('n'):
            params = parse_run_name(item)
            if params:
                # Get available plots - check all files in the directory
                plots = []
                try:
                    for filename in os.listdir(item_path):
                        if filename.endswith('.png'):
                            plot_path = os.path.join(item_path, filename)
                            plots.append({
                                'name': filename,
                                'path': f'{item}/{filename}',
                                'mtime': os.path.getmtime(plot_path)
                            })
                except Exception as e:
                    print(f"Error reading plots from {item_path}: {e}")


                # Read metrics if available
                metrics = {}
                metrics_path = os.path.join(item_path, 'metrics.txt')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        for line in f:
                            if ':' in line:
                                key, val = line.strip().split(':', 1)
                                try:
                                    metrics[key.strip()] = float(val.strip())
                                except:
                                    metrics[key.strip()] = val.strip()

                runs.append({
                    'name': item,
                    'params': params,
                    'plots': plots,
                    'metrics': metrics,
                    'last_updated': max([p['mtime'] for p in plots]) if plots else 0
                })

    return sorted(runs, key=lambda x: x['name'])

def get_plot_categories():
    """Return categorized list of available plot types."""
    return {
        'Training': [
            'training_progress.png',
            'training_summary_2panel.png'
        ],
        'Scattering': [
            'kappa_recovery.png',
            'eigenvectors_squared.png',
            'eigenvectors.png'
        ],
        'Trained Model': [
            'trained_deriv.png',
            'trained_res.png',
            'trained_iom.png'
        ],
        'Error Analysis': [
            'error_deriv.png',
            'error_res.png'
        ],
        'Reference': [
            'analytic_deriv.png',
            'analytic_res.png',
            'pretrain_deriv.png',
            'pretrain_res.png'
        ]
    }

@app.route('/')
def index():
    """Main viewer page."""
    return render_template('sweep_viewer.html')

@app.after_request
def add_no_cache_headers(response):
    """Add headers to disable caching for API endpoints."""
    if request.path.startswith('/api/') or request.path.startswith('/sweep_results/'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

@app.route('/api/runs')
def api_runs():
    """API endpoint to get all runs with optional filtering."""
    runs = get_available_runs()

    # Apply filters from query parameters
    n_solitons = request.args.get('n_solitons', type=int)
    min_kappa = request.args.get('min_kappa', type=float)
    max_kappa = request.args.get('max_kappa', type=float)

    import math

    filtered_runs = []
    for run in runs:
        params = run['params']

        # Filter by n_solitons
        if n_solitons is not None and params['n_solitons'] != n_solitons:
            continue

        # Filter by kappa range
        if min_kappa is not None and max(params['kappas']) < min_kappa:
            continue
        if max_kappa is not None and min(params['kappas']) > max_kappa:
            continue

        # Clean metrics: replace NaN/Inf with None (null in JSON)
        cleaned_metrics = {}
        for key, val in run['metrics'].items():
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                cleaned_metrics[key] = None
            else:
                cleaned_metrics[key] = val
        run['metrics'] = cleaned_metrics

        filtered_runs.append(run)

    return jsonify(filtered_runs)

@app.route('/api/plot_categories')
def api_plot_categories():
    """API endpoint to get plot categories."""
    return jsonify(get_plot_categories())

@app.route('/sweep_results/<path:filename>')
def serve_plot(filename):
    """Serve plot images from sweep_results directory."""
    return send_from_directory(SWEEP_DIR, filename)

@app.route('/api/compare')
def api_compare():
    """API endpoint to compare a specific plot across runs."""
    import math

    plot_name = request.args.get('plot')
    runs = get_available_runs()

    comparison = []
    for run in runs:
        for plot in run['plots']:
            if plot['name'] == plot_name:
                # Clean metrics: replace NaN/Inf with None (null in JSON)
                cleaned_metrics = {}
                for key, val in run['metrics'].items():
                    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                        cleaned_metrics[key] = None
                    else:
                        cleaned_metrics[key] = val

                comparison.append({
                    'run_name': run['name'],
                    'params': run['params'],
                    'plot_path': plot['path'],
                    'metrics': cleaned_metrics
                })
                break

    return jsonify(comparison)

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Parameter Sweep Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }
        .header {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        h1 { color: #4CAF50; margin-bottom: 10px; }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        label {
            color: #aaa;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        select, input {
            padding: 8px;
            background: #333;
            border: 1px solid #555;
            border-radius: 4px;
            color: #e0e0e0;
            font-size: 14px;
        }
        button {
            padding: 10px 20px;
            background: #4CAF50;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.3s;
        }
        button:hover { background: #45a049; }
        button.secondary {
            background: #666;
        }
        button.secondary:hover { background: #555; }
        .view-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .view-tab {
            padding: 10px 20px;
            background: #2a2a2a;
            border: none;
            border-radius: 4px;
            color: #aaa;
            cursor: pointer;
            transition: all 0.3s;
        }
        .view-tab.active {
            background: #4CAF50;
            color: white;
        }
        .content { display: none; }
        .content.active { display: block; }
        .runs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }
        .run-card {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        .run-card:hover { transform: translateY(-2px); }
        .run-header {
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #444;
        }
        .run-name {
            font-weight: bold;
            color: #4CAF50;
            font-size: 14px;
            margin-bottom: 5px;
        }
        .run-params {
            font-size: 12px;
            color: #999;
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 5px 10px;
        }
        .param-label { color: #666; }
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
        }
        .plot-thumb {
            position: relative;
            cursor: pointer;
            border-radius: 4px;
            overflow: hidden;
            background: #1a1a1a;
        }
        .plot-thumb img {
            width: 100%;
            height: 120px;
            object-fit: cover;
            transition: opacity 0.3s;
        }
        .plot-thumb:hover img { opacity: 0.8; }
        .plot-label {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 5px;
            background: rgba(0,0,0,0.8);
            font-size: 11px;
            text-align: center;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
        }
        .comparison-card {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .comparison-card img {
            width: 100%;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .metrics {
            font-size: 11px;
            color: #888;
            margin-top: 10px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 3px;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            padding: 40px;
            overflow: auto;
        }
        .modal.active { display: flex; justify-content: center; align-items: center; }
        .modal img {
            max-width: 100%;
            max-height: 90vh;
            border-radius: 8px;
        }
        .modal-close {
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 30px;
            color: white;
            cursor: pointer;
        }
        .status {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            margin-left: 10px;
        }
        .status.live { background: #4CAF50; color: white; }
        .status.complete { background: #666; color: white; }
        .live-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4CAF50;
            animation: pulse 2s infinite;
            margin-right: 5px;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Parameter Sweep Viewer</h1>
        <div class="controls">
            <div class="control-group">
                <label>Number of Solitons</label>
                <select id="filterNSolitons">
                    <option value="">All</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="5">5</option>
                    <option value="7">7</option>
                </select>
            </div>
            <div class="control-group">
                <label>Min Kappa</label>
                <input type="number" id="filterMinKappa" step="0.1" placeholder="Any">
            </div>
            <div class="control-group">
                <label>Max Kappa</label>
                <input type="number" id="filterMaxKappa" step="0.1" placeholder="Any">
            </div>
            <div class="control-group">
                <label>Auto-refresh</label>
                <select id="autoRefresh">
                    <option value="2000">2 seconds</option>
                    <option value="5000">5 seconds</option>
                    <option value="10000">10 seconds</option>
                    <option value="0">Off</option>
                </select>
            </div>
            <div class="control-group">
                <label>&nbsp;</label>
                <button onclick="applyFilters()">Apply Filters</button>
            </div>
        </div>
    </div>

    <div class="view-tabs">
        <button class="view-tab active" onclick="switchView('gallery')">📊 Gallery View</button>
        <button class="view-tab" onclick="switchView('compare')">🔀 Compare View</button>
    </div>

    <div id="galleryView" class="content active">
        <div id="runsContainer" class="runs-grid"></div>
    </div>

    <div id="compareView" class="content">
        <div class="control-group" style="max-width: 400px; margin-bottom: 20px;">
            <label>Select Plot to Compare</label>
            <select id="comparePlot" onchange="loadComparison()"></select>
        </div>
        <div id="comparisonContainer" class="comparison-grid"></div>
    </div>

    <div id="modal" class="modal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img id="modalImage" src="">
    </div>

    <script>
        let refreshInterval = null;
        let currentView = 'gallery';

        function switchView(view) {
            currentView = view;
            document.querySelectorAll('.view-tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.content').forEach(content => content.classList.remove('active'));

            event.target.classList.add('active');
            document.getElementById(view + 'View').classList.add('active');

            if (view === 'compare') {
                loadPlotCategories();
            }
        }

        function loadRuns() {
            const params = new URLSearchParams();
            const nSolitons = document.getElementById('filterNSolitons').value;
            const minKappa = document.getElementById('filterMinKappa').value;
            const maxKappa = document.getElementById('filterMaxKappa').value;

            if (nSolitons) params.append('n_solitons', nSolitons);
            if (minKappa) params.append('min_kappa', minKappa);
            if (maxKappa) params.append('max_kappa', maxKappa);

            fetch('/api/runs?' + params.toString())
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
                    return r.json();
                })
                .then(runs => {
                    console.log(`Loaded ${runs.length} runs`);
                    const container = document.getElementById('runsContainer');
                    if (runs.length === 0) {
                        container.innerHTML = '<div style="padding: 40px; text-align: center; color: #999;">No runs found. Check that sweep_results/ directory exists and contains run directories.</div>';
                        return;
                    }
                    container.innerHTML = runs.map(run => {
                        const isRecent = (Date.now() / 1000 - run.last_updated) < 60;
                        const status = isRecent ?
                            '<span class="status live"><span class="live-indicator"></span>Live</span>' :
                            '<span class="status complete">Complete</span>';

                        return `
                            <div class="run-card">
                                <div class="run-header">
                                    <div class="run-name">${run.name} ${status}</div>
                                    <div class="run-params">
                                        <span class="param-label">N:</span><span>${run.params.n_solitons}</span>
                                        <span class="param-label">κ:</span><span>${run.params.kappas.map(k => k.toFixed(2)).join(', ')}</span>
                                        <span class="param-label">x₀:</span><span>${run.params.x0s.map(x => x.toFixed(1)).join(', ')}</span>
                                    </div>
                                </div>
                                <div class="plot-grid">
                                    ${run.plots.map(plot => `
                                        <div class="plot-thumb" onclick="showImage('/sweep_results/${plot.path}')">
                                            <img src="/sweep_results/${plot.path}" alt="${plot.name}">
                                            <div class="plot-label">${plot.name.replace('.png', '').replace(/_/g, ' ')}</div>
                                        </div>
                                    `).join('')}
                                </div>
                                ${Object.keys(run.metrics).length > 0 ? `
                                    <div class="metrics">
                                        ${Object.entries(run.metrics).slice(0, 6).map(([k, v]) =>
                                            `<div>${k}: ${typeof v === 'number' ? v.toExponential(2) : v}</div>`
                                        ).join('')}
                                    </div>
                                ` : ''}
                            </div>
                        `;
                    }).join('');
                })
                .catch(err => {
                    console.error('Error loading runs:', err);
                    document.getElementById('runsContainer').innerHTML =
                        `<div style="padding: 40px; text-align: center; color: #f44;">
                            Error loading runs: ${err.message}<br>
                            Check browser console for details.
                        </div>`;
                });
        }

        function loadPlotCategories() {
            fetch('/api/plot_categories')
                .then(r => r.json())
                .then(categories => {
                    const select = document.getElementById('comparePlot');
                    select.innerHTML = Object.entries(categories).map(([category, plots]) =>
                        `<optgroup label="${category}">
                            ${plots.map(plot => `<option value="${plot}">${plot}</option>`).join('')}
                        </optgroup>`
                    ).join('');
                    loadComparison();
                });
        }

        function loadComparison() {
            const plot = document.getElementById('comparePlot').value;
            if (!plot) return;

            fetch('/api/compare?plot=' + encodeURIComponent(plot))
                .then(r => r.json())
                .then(comparison => {
                    const container = document.getElementById('comparisonContainer');
                    container.innerHTML = comparison.map(item => `
                        <div class="comparison-card">
                            <img src="/sweep_results/${item.plot_path}" alt="${item.run_name}" onclick="showImage('/sweep_results/${item.plot_path}')">
                            <div class="run-name">${item.run_name}</div>
                            <div class="run-params">
                                N=${item.params.n_solitons},
                                κ=[${item.params.kappas.map(k => k.toFixed(2)).join(', ')}],
                                x₀=[${item.params.x0s.map(x => x.toFixed(1)).join(', ')}]
                            </div>
                            ${Object.keys(item.metrics).length > 0 ? `
                                <div class="metrics">
                                    ${Object.entries(item.metrics).slice(0, 4).map(([k, v]) =>
                                        `<div>${k}: ${typeof v === 'number' ? v.toExponential(2) : v}</div>`
                                    ).join('')}
                                </div>
                            ` : ''}
                        </div>
                    `).join('');
                });
        }

        function applyFilters() {
            loadRuns();
            setupAutoRefresh();
        }

        function setupAutoRefresh() {
            if (refreshInterval) clearInterval(refreshInterval);

            const interval = parseInt(document.getElementById('autoRefresh').value);
            if (interval > 0) {
                refreshInterval = setInterval(() => {
                    if (currentView === 'gallery') {
                        loadRuns();
                    } else {
                        loadComparison();
                    }
                }, interval);
            }
        }

        function showImage(src) {
            document.getElementById('modalImage').src = src;
            document.getElementById('modal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }

        // Initialize
        loadRuns();
        setupAutoRefresh();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Create templates directory and save HTML
    os.makedirs('templates', exist_ok=True)
    with open('templates/sweep_viewer.html', 'w') as f:
        f.write(HTML_TEMPLATE)

    import sys

    # Allow custom port via command line argument
    port = 5001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default port {port}")

    print("=" * 80)
    print("Parameter Sweep Viewer")
    print("=" * 80)
    print(f"\n📂 Watching directory: {SWEEP_DIR}/")
    print(f"🌐 Server starting at: http://localhost:{port}")
    print("\nFeatures:")
    print("  • Real-time monitoring of validation runs")
    print("  • Filter by n_solitons, kappa range")
    print("  • Gallery view: see all runs and plots")
    print("  • Compare view: compare same plot across runs")
    print("  • Auto-refresh (2s, 5s, 10s intervals)")
    print("\nPress Ctrl+C to stop\n")

    app.run(debug=True, host='0.0.0.0', port=port)
