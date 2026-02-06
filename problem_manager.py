#!/usr/bin/env python3
"""
CodeEvolve Problem Manager - Web UI for managing evolution problems
"""
import os
import subprocess
import json
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, Response
import threading
import queue

app = Flask(__name__)

REPO_ROOT = Path(__file__).parent
PROBLEMS_DIR = REPO_ROOT / "problems"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>CodeEvolve Problem Manager</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            font-size: 32px;
            margin-bottom: 30px;
            color: #58a6ff;
            border-bottom: 2px solid #21262d;
            padding-bottom: 15px;
        }
        .problem-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .problem-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
            transition: all 0.2s;
        }
        .problem-card:hover {
            border-color: #58a6ff;
            box-shadow: 0 0 10px rgba(88, 166, 255, 0.2);
        }
        .problem-card.selected {
            border-color: #58a6ff;
            background: #0d419d20;
        }
        .problem-name {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #58a6ff;
            cursor: pointer;
        }
        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 15px;
        }
        button {
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        button:hover {
            background: #30363d;
            border-color: #58a6ff;
        }
        button.primary {
            background: #238636;
            border-color: #238636;
        }
        button.primary:hover {
            background: #2ea043;
        }
        button.danger {
            background: #da3633;
            border-color: #da3633;
        }
        button.danger:hover {
            background: #f85149;
        }
        .runs-list {
            margin-top: 10px;
            font-size: 13px;
            color: #8b949e;
        }
        .runs-list span {
            display: inline-block;
            background: #21262d;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 4px;
            border: 1px solid #30363d;
        }
        .output-panel {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
            display: none;
        }
        .output-panel.active { display: block; }
        .output-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #30363d;
        }
        .output-content {
            background: #010409;
            padding: 15px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            max-height: 600px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .loading {
            color: #58a6ff;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .input-group {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            align-items: center;
        }
        input[type="text"], input[type="number"] {
            background: #0d1117;
            border: 1px solid #30363d;
            color: #c9d1d9;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            flex: 1;
        }
        input:focus {
            outline: none;
            border-color: #58a6ff;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-running { background: #1f6feb; color: white; }
        .status-success { background: #238636; color: white; }
        .status-error { background: #da3633; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 CodeEvolve Problem Manager</h1>

        <div class="problem-grid" id="problemGrid"></div>

        <div class="output-panel" id="outputPanel">
            <div class="output-header">
                <h2 id="outputTitle">Output</h2>
                <button onclick="closeOutput()">Close</button>
            </div>
            <div class="output-content" id="outputContent"></div>
        </div>
    </div>

    <script>
        let selectedProblem = null;
        let outputEventSource = null;

        async function loadProblems() {
            const res = await fetch('/api/problems');
            const problems = await res.json();

            const grid = document.getElementById('problemGrid');
            grid.innerHTML = '';

            problems.forEach(problem => {
                const card = document.createElement('div');
                card.className = 'problem-card';
                card.innerHTML = `
                    <div class="problem-name" onclick="selectProblem('${problem.name}')">${problem.name}</div>
                    <div class="runs-list">
                        ${problem.runs.length > 0 ?
                            'Runs: ' + problem.runs.map(r => `<span>${r}</span>`).join('') :
                            '<span>No runs yet</span>'}
                    </div>
                    <div class="actions">
                        <button class="primary" onclick="runCommand('${problem.name}', 'run', ['--next'])">▶ Run Next</button>
                        <button onclick="runCommand('${problem.name}', 'analyze')">📊 Analyze</button>
                        ${problem.runs.length > 0 ? `
                            <button onclick="showVizOptions('${problem.name}', ${JSON.stringify(problem.runs)})">👁 Visualize</button>
                            <button onclick="showTailOptions('${problem.name}', ${JSON.stringify(problem.runs)})">📜 Tail Logs</button>
                        ` : ''}
                        <button onclick="runCommand('${problem.name}', 'ls')">📋 List Runs</button>
                        <button onclick="runCommand('${problem.name}', 'winner')">🏆 Winner</button>
                    </div>
                `;
                grid.appendChild(card);
            });
        }

        function selectProblem(name) {
            selectedProblem = name;
            document.querySelectorAll('.problem-card').forEach(card => {
                card.classList.toggle('selected',
                    card.querySelector('.problem-name').textContent === name);
            });
        }

        async function runCommand(problem, command, args = []) {
            showOutput(`Running: ${problem} ${command} ${args.join(' ')}`, true);

            if (outputEventSource) {
                outputEventSource.close();
            }

            const params = new URLSearchParams({
                problem: problem,
                command: command,
                args: JSON.stringify(args)
            });

            outputEventSource = new EventSource('/api/run?' + params);

            outputEventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.done) {
                    appendOutput(`\n\n✓ Command completed with exit code: ${data.code}`);
                    outputEventSource.close();
                } else {
                    appendOutput(data.output);
                }
            };

            outputEventSource.onerror = () => {
                appendOutput('\n\n✗ Connection error');
                outputEventSource.close();
            };
        }

        function showVizOptions(problem, runs) {
            const run = prompt(`Which run to visualize? (${runs.join(', ')})`, runs[runs.length - 1]);
            if (run) {
                const island = prompt('Which island?', '0');
                runCommand(problem, 'viz', [run, island || '0']);
            }
        }

        function showTailOptions(problem, runs) {
            const run = prompt(`Which run to tail? (${runs.join(', ')})`, runs[runs.length - 1]);
            if (run) {
                const island = prompt('Which island?', '0');
                runCommand(problem, 'tail', [run, island || '0']);
            }
        }

        function showOutput(title, loading = false) {
            document.getElementById('outputTitle').innerHTML =
                title + (loading ? ' <span class="loading">●</span>' : '');
            document.getElementById('outputContent').textContent = '';
            document.getElementById('outputPanel').classList.add('active');
        }

        function appendOutput(text) {
            const content = document.getElementById('outputContent');
            content.textContent += text;
            content.scrollTop = content.scrollHeight;
        }

        function closeOutput() {
            if (outputEventSource) {
                outputEventSource.close();
            }
            document.getElementById('outputPanel').classList.remove('active');
        }

        // Load problems on startup
        loadProblems();

        // Refresh every 10 seconds
        setInterval(loadProblems, 10000);
    </script>
</body>
</html>
"""

def get_problems():
    """Get list of all problems with their runs"""
    problems = []

    for item in sorted(PROBLEMS_DIR.iterdir()):
        if item.is_dir() and not item.name.startswith('.') and item.name != 'problem_template':
            # Check for shell script
            shell_script = None
            for ext in [f'{item.name}.sh', 'run.sh']:
                script_path = item / ext
                if script_path.exists():
                    shell_script = script_path
                    break

            # Get runs for this problem
            runs = []
            exp_dir = EXPERIMENTS_DIR / item.name
            if exp_dir.exists():
                runs = sorted([
                    d.name for d in exp_dir.iterdir()
                    if d.is_dir() and d.name.startswith('run')
                ])

            problems.append({
                'name': item.name,
                'path': str(item),
                'script': str(shell_script) if shell_script else None,
                'runs': runs
            })

    return problems

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/problems')
def api_problems():
    return jsonify(get_problems())

@app.route('/api/run')
def api_run():
    problem = request.args.get('problem')
    command = request.args.get('command')
    args = json.loads(request.args.get('args', '[]'))

    def generate():
        # Find the problem's shell script
        problem_dir = PROBLEMS_DIR / problem
        script = None

        for script_name in [f'{problem}.sh', 'run.sh']:
            script_path = problem_dir / script_name
            if script_path.exists():
                script = script_path
                break

        if not script:
            yield json.dumps({'output': f'No shell script found for {problem}\n', 'done': False}) + '\n'
            yield json.dumps({'done': True, 'code': 1}) + '\n'
            return

        # Build command
        cmd = ['bash', str(script), command] + args

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(problem_dir)
            )

            # Stream output
            for line in process.stdout:
                yield f'data: {json.dumps({"output": line, "done": False})}\n\n'

            process.wait()
            yield f'data: {json.dumps({"done": True, "code": process.returncode})}\n\n'

        except Exception as e:
            yield f'data: {json.dumps({"output": f"Error: {str(e)}\n", "done": False})}\n\n'
            yield f'data: {json.dumps({"done": True, "code": 1})}\n\n'

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    print(f"""
╔══════════════════════════════════════════╗
║  CodeEvolve Problem Manager Started!     ║
╚══════════════════════════════════════════╝

🌐 Open in browser: http://localhost:5000

Available actions:
  • Run Next - Start a new evolution run
  • Analyze - Analyze all runs
  • Visualize - View results for a run
  • Tail Logs - Watch live logs
  • List Runs - Show all runs
  • Winner - Find best solution

Press Ctrl+C to stop
""")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
