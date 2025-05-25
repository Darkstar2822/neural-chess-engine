"""
Training Dashboard Web Interface
Provides a web-based dashboard for viewing training visualizations and progress
"""

import os
import json
import glob
from flask import Flask, render_template, jsonify, request, send_file
from datetime import datetime
import sys
sys.path.append('..')
from utils.training_visualizer import TrainingVisualizer
from utils.training_monitor import create_monitor

app = Flask(__name__, template_folder='templates', static_folder='static')

class TrainingDashboard:
    """Web dashboard for training visualization"""
    
    def __init__(self, training_logs_dir: str = "training_logs"):
        self.logs_dir = training_logs_dir
        os.makedirs(training_logs_dir, exist_ok=True)
    
    def get_training_sessions(self):
        """Get list of all training sessions"""
        sessions = []
        
        # Look for metadata files
        metadata_files = glob.glob(os.path.join(self.logs_dir, "*_metadata.json"))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                session_id = metadata.get('session_id', 'unknown')
                training_type = metadata.get('training_type', 'unknown')
                start_time = metadata.get('start_time', 'unknown')
                duration = metadata.get('total_duration', 0)
                
                # Look for associated files
                html_files = glob.glob(os.path.join(self.logs_dir, f"{session_id}*.html"))
                report_file = os.path.join(self.logs_dir, f"{session_id}_report.md")
                
                sessions.append({
                    'session_id': session_id,
                    'training_type': training_type,
                    'start_time': start_time,
                    'duration': duration,
                    'duration_minutes': duration / 60 if duration else 0,
                    'has_visualization': len(html_files) > 0,
                    'has_report': os.path.exists(report_file),
                    'metadata_file': metadata_file,
                    'html_files': html_files
                })
                
            except Exception as e:
                print(f"Error reading metadata from {metadata_file}: {e}")
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x['start_time'], reverse=True)
        return sessions
    
    def get_session_details(self, session_id: str):
        """Get detailed information about a specific session"""
        metadata_file = os.path.join(self.logs_dir, f"{session_id}_metadata.json")
        
        if not os.path.exists(metadata_file):
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Get associated files
            html_files = glob.glob(os.path.join(self.logs_dir, f"{session_id}*.html"))
            report_file = os.path.join(self.logs_dir, f"{session_id}_report.md")
            
            report_content = ""
            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    report_content = f.read()
            
            return {
                'metadata': metadata,
                'html_files': html_files,
                'report_content': report_content,
                'has_files': len(html_files) > 0
            }
            
        except Exception as e:
            print(f"Error reading session details: {e}")
            return None

# Global dashboard instance
dashboard = TrainingDashboard("../training_logs")

@app.route('/')
def index():
    """Main dashboard page"""
    sessions = dashboard.get_training_sessions()
    return render_template('training_dashboard.html', sessions=sessions)

@app.route('/api/sessions')
def api_sessions():
    """API endpoint for training sessions"""
    sessions = dashboard.get_training_sessions()
    return jsonify(sessions)

@app.route('/api/session/<session_id>')
def api_session_details(session_id):
    """API endpoint for session details"""
    details = dashboard.get_session_details(session_id)
    if details:
        return jsonify(details)
    else:
        return jsonify({'error': 'Session not found'}), 404

@app.route('/visualization/<session_id>')
def view_visualization(session_id):
    """View visualization for a specific session"""
    details = dashboard.get_session_details(session_id)
    
    if not details or not details['has_files']:
        return "Visualization not found", 404
    
    # Get the most recent HTML file for this session
    html_files = sorted(details['html_files'], key=os.path.getmtime, reverse=True)
    latest_html = html_files[0]
    
    return send_file(latest_html)

@app.route('/report/<session_id>')
def view_report(session_id):
    """View report for a specific session"""
    details = dashboard.get_session_details(session_id)
    
    if not details:
        return "Report not found", 404
    
    return render_template('training_report.html', 
                         session_id=session_id,
                         report_content=details['report_content'],
                         metadata=details['metadata'])

@app.route('/api/live_sessions')
def api_live_sessions():
    """Get currently active training sessions"""
    # This would need to be integrated with actual training processes
    # For now, return empty list
    return jsonify([])

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start monitoring a new training session"""
    data = request.get_json()
    training_type = data.get('training_type', 'standard')
    params = data.get('parameters', {})
    
    try:
        monitor = create_monitor(training_type, save_dir="../training_logs")
        monitor.start_monitoring(params)
        
        return jsonify({
            'success': True,
            'session_id': monitor.session_id,
            'message': f'Started monitoring {training_type} training'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Create templates directory and basic templates
def create_templates():
    """Create template files for the dashboard"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    # Main dashboard template
    dashboard_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Neural Chess Training Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .stat-card h3 {
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }
        .sessions-container {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .session-card {
            background: rgba(255,255,255,0.1);
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
            transition: transform 0.2s;
        }
        .session-card:hover {
            transform: translateX(5px);
        }
        .session-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .session-type {
            background: #4CAF50;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .session-actions {
            display: flex;
            gap: 10px;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            color: white;
            font-weight: bold;
            transition: background-color 0.2s;
        }
        .btn-primary {
            background: #2196F3;
        }
        .btn-primary:hover {
            background: #1976D2;
        }
        .btn-secondary {
            background: #FF9800;
        }
        .btn-secondary:hover {
            background: #F57C00;
        }
        .empty-state {
            text-align: center;
            padding: 40px;
            color: rgba(255,255,255,0.7);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Neural Chess Training Dashboard</h1>
        <p>Monitor and visualize your chess AI training progress</p>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <h3>📊 Total Sessions</h3>
            <div class="stat-value">{{ sessions|length }}</div>
        </div>
        <div class="stat-card">
            <h3>🧬 Training Types</h3>
            <div class="stat-value">{{ sessions|map(attribute='training_type')|unique|list|length }}</div>
        </div>
        <div class="stat-card">
            <h3>⏱️ Total Training Time</h3>
            <div class="stat-value">{{ "%.1f"|format(sessions|sum(attribute='duration_minutes')) }}m</div>
        </div>
        <div class="stat-card">
            <h3>🔥 Active Sessions</h3>
            <div class="stat-value">0</div>
        </div>
    </div>
    
    <div class="sessions-container">
        <h2>🏆 Training Sessions</h2>
        
        {% if sessions %}
            {% for session in sessions %}
            <div class="session-card">
                <div class="session-header">
                    <div>
                        <strong>{{ session.session_id }}</strong>
                        <span class="session-type">{{ session.training_type }}</span>
                    </div>
                    <div class="session-actions">
                        {% if session.has_visualization %}
                        <a href="/visualization/{{ session.session_id }}" class="btn btn-primary" target="_blank">📈 View Charts</a>
                        {% endif %}
                        {% if session.has_report %}
                        <a href="/report/{{ session.session_id }}" class="btn btn-secondary" target="_blank">📄 View Report</a>
                        {% endif %}
                    </div>
                </div>
                <div>
                    <small>
                        Started: {{ session.start_time[:19] if session.start_time != 'unknown' else 'Unknown' }} | 
                        Duration: {{ "%.1f"|format(session.duration_minutes) }} minutes
                    </small>
                </div>
            </div>
            {% endfor %}
        {% else %}
            <div class="empty-state">
                <h3>🚀 No training sessions yet</h3>
                <p>Start training your chess AI to see visualizations here!</p>
            </div>
        {% endif %}
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => {
            window.location.reload();
        }, 30000);
    </script>
</body>
</html>'''
    
    with open(os.path.join(template_dir, 'training_dashboard.html'), 'w') as f:
        f.write(dashboard_template)
    
    # Report template
    report_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📄 Training Report - {{ session_id }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.6;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .metadata {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .back-btn {
            display: inline-block;
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Dashboard</a>
        
        <div class="header">
            <h1>📄 Training Report</h1>
            <h2>{{ session_id }}</h2>
        </div>
        
        <div class="metadata">
            <h3>📊 Session Metadata</h3>
            <ul>
                <li><strong>Training Type:</strong> {{ metadata.training_type }}</li>
                <li><strong>Start Time:</strong> {{ metadata.start_time }}</li>
                <li><strong>Duration:</strong> {{ "%.1f"|format(metadata.total_duration/60 if metadata.total_duration else 0) }} minutes</li>
            </ul>
        </div>
        
        <div class="report-content">
            <pre>{{ report_content }}</pre>
        </div>
    </div>
</body>
</html>'''
    
    with open(os.path.join(template_dir, 'training_report.html'), 'w') as f:
        f.write(report_template)

def run_dashboard(host='127.0.0.1', port=5004, debug=True):
    """Run the training dashboard"""
    create_templates()
    print(f"🎯 Starting Training Dashboard at http://{host}:{port}")
    print("📊 View your training visualizations and progress")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_dashboard()