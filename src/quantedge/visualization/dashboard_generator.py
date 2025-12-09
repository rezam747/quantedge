"""
Dashboard and visualization generation module.
Creates interactive plots and HTML dashboards.
"""

import plotly.graph_objects as go
from datetime import datetime
import os
import subprocess
import platform


class DashboardGenerator:
    """
    Generates interactive dashboards and visualizations.
    """
    
    def __init__(self, data, symbol="BTC-USD"):
        """
        Initialize DashboardGenerator.
        
        Args:
            data (pd.DataFrame): Data with features
            symbol (str): Trading symbol
        """
        self.data = data
        self.symbol = symbol
    
    def create_trading_signals_plot(self, train_idx, test_idx, output_file):
        """
        Create trading signals plot with green dots for training and blue dots for testing.
        
        Args:
            train_idx: Training data indices
            test_idx: Test data indices
            output_file (str): Output HTML file path
        """
        df = self.data.copy()
        
        fig = go.Figure()
        
        # Plot price line
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Close'], 
            mode='lines', 
            name='Close Price', 
            line=dict(color='black', width=1)
        ))
        
        # Plot training signals with target=1 (green dots)
        train_target_1_mask = (df.index.isin(train_idx)) & (df['signal_labels'] == 1)
        fig.add_trace(go.Scatter(
            x=df.index[train_target_1_mask],
            y=df['Close'][train_target_1_mask],
            mode='markers',
            marker=dict(color='green', size=8, symbol='circle'),
            name='Training Target=1'
        ))
        
        # Plot test signals with target=1 (blue dots)
        test_target_1_mask = (df.index.isin(test_idx)) & (df['signal_labels'] == 1)
        fig.add_trace(go.Scatter(
            x=df.index[test_target_1_mask],
            y=df['Close'][test_target_1_mask],
            mode='markers',
            marker=dict(color='blue', size=8, symbol='circle'),
            name='Test Target=1'
        ))
        
        fig.update_layout(
            title=f"Random Forest Trading Signals - {self.symbol}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode='x unified'
        )
        
        fig.write_html(output_file)
        print(f"Exported trading signals plot to {output_file}")
    
    def create_model_info_html(self, train_metrics, test_metrics, model_params, 
                               num_train_samples, num_test_samples, 
                               num_train_target_1, num_test_target_1,
                               num_train_signals, num_test_signals,
                               cm, class_report, stop_loss_pct, target_pct,
                               output_file):
        """
        Create model information HTML page.
        
        Args:
            train_metrics (dict): Training metrics
            test_metrics (dict): Test metrics
            model_params (dict): Model parameters
            num_train_samples (int): Number of training samples
            num_test_samples (int): Number of test samples
            num_train_target_1 (int): Number of training samples with target=1
            num_test_target_1 (int): Number of test samples with target=1
            num_train_signals (int): Number of buy signals in training
            num_test_signals (int): Number of buy signals in testing
            cm: Confusion matrix
            class_report (dict): Classification report
            stop_loss_pct (float): Stop loss percentage
            target_pct (float): Target percentage
            output_file (str): Output HTML file path
        """
        model_info_html = f"""
        <!DOCTYPE html>
        <html lang='en'>
        <head>
            <meta charset='UTF-8'>
            <title>Random Forest Model Information</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f8f8f8; padding: 20px; }}
                h1, h2 {{ color: #222; }}
                .info-section {{ background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ font-size: 18px; font-weight: bold; color: #4CAF50; }}
            </style>
        </head>
        <body>
            <h1>Random Forest Model Information</h1>
            
            <div class="info-section">
                <h2>Model Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Model Type</td><td>Random Forest Classifier</td></tr>
                    <tr><td>Number of Estimators</td><td>{model_params.get('n_estimators', 'N/A')}</td></tr>
                    <tr><td>Max Depth</td><td>{model_params.get('max_depth', 'N/A')}</td></tr>
                    <tr><td>Min Samples Split</td><td>{model_params.get('min_samples_split', 'N/A')}</td></tr>
                    <tr><td>Min Samples Leaf</td><td>{model_params.get('min_samples_leaf', 'N/A')}</td></tr>
                    <tr><td>Class Weight</td><td>{model_params.get('class_weight', 'N/A')}</td></tr>
                    <tr><td>Random State</td><td>{model_params.get('random_state', 'N/A')}</td></tr>
                </table>
            </div>
            
            <div class="info-section">
                <h2>Training Data Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Training Samples</td><td>{num_train_samples}</td></tr>
                    <tr><td>Training Samples with Target=1</td><td>{num_train_target_1} ({num_train_target_1/num_train_samples*100:.2f}%)</td></tr>
                    <tr><td>Training Accuracy</td><td class="metric">{train_metrics['accuracy']:.4f}</td></tr>
                    <tr><td>Training Precision</td><td class="metric">{train_metrics['precision']:.4f}</td></tr>
                    <tr><td>Training Recall</td><td class="metric">{train_metrics['recall']:.4f}</td></tr>
                    <tr><td>Buy Signals Generated (Training)</td><td>{num_train_signals}</td></tr>
                </table>
            </div>
            
            <div class="info-section">
                <h2>Test Data Performance</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Test Samples</td><td>{num_test_samples}</td></tr>
                    <tr><td>Test Samples with Target=1</td><td>{num_test_target_1} ({num_test_target_1/num_test_samples*100:.2f}%)</td></tr>
                    <tr><td>Test Accuracy</td><td class="metric">{test_metrics['accuracy']:.4f}</td></tr>
                    <tr><td>Test Precision</td><td class="metric">{test_metrics['precision']:.4f}</td></tr>
                    <tr><td>Test Recall</td><td class="metric">{test_metrics['recall']:.4f}</td></tr>
                    <tr><td>Buy Signals Generated (Test)</td><td>{num_test_signals}</td></tr>
                </table>
            </div>
            
            <div class="info-section">
                <h2>Confusion Matrix (Test Data)</h2>
                <table>
                    <tr><th></th><th>Predicted: 0</th><th>Predicted: 1</th></tr>
                    <tr><th>Actual: 0</th><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr>
                    <tr><th>Actual: 1</th><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr>
                </table>
            </div>
            
            <div class="info-section">
                <h2>Classification Report (Test Data)</h2>
                <table>
                    <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
                    <tr><td>Class 0</td><td>{class_report['0']['precision']:.4f}</td><td>{class_report['0']['recall']:.4f}</td><td>{class_report['0']['f1-score']:.4f}</td><td>{int(class_report['0']['support'])}</td></tr>
                    <tr><td>Class 1</td><td>{class_report['1']['precision']:.4f}</td><td>{class_report['1']['recall']:.4f}</td><td>{class_report['1']['f1-score']:.4f}</td><td>{int(class_report['1']['support'])}</td></tr>
                    <tr><td><strong>Weighted Avg</strong></td><td>{class_report['weighted avg']['precision']:.4f}</td><td>{class_report['weighted avg']['recall']:.4f}</td><td>{class_report['weighted avg']['f1-score']:.4f}</td><td>{int(class_report['weighted avg']['support'])}</td></tr>
                </table>
            </div>
            
            <div class="info-section">
                <h2>Trading Parameters</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Symbol</td><td>{self.symbol}</td></tr>
                    <tr><td>Stop Loss Percentage</td><td>{stop_loss_pct}%</td></tr>
                    <tr><td>Target Percentage</td><td>{target_pct}%</td></tr>
                    <tr><td>Data Source</td><td>Yahoo Finance</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(model_info_html)
        print(f"Exported model information to {output_file}")
    
    def create_training_data_table(self, X_train, y_train, predictions, data, output_file):
        """
        Create training data table with predictions.
        
        Args:
            X_train: Training features
            y_train: Training labels
            predictions: Model predictions on training data
            data: Full dataset with features
            output_file (str): Output HTML file path
        """
        # Create DataFrame with training data
        train_data = data.loc[X_train.index].copy()
        train_data['predicted_signal_labels'] = predictions
        
        train_data.to_html(output_file)
        print(f"Exported training data table to {output_file}")
    
    def create_testing_data_table(self, X_test, y_test, predictions, data, output_file):
        """
        Create testing data table with predictions.
        
        Args:
            X_test: Test features
            y_test: Test labels
            predictions: Model predictions on test data
            data: Full dataset with features
            output_file (str): Output HTML file path
        """
        # Create DataFrame with test data
        test_data = data.loc[X_test.index].copy()
        test_data['predicted_signal_labels'] = predictions
        
        test_data.to_html(output_file)
        print(f"Exported testing data table to {output_file}")
    
    def create_dashboard(self, timestamp, output_file):
        """
        Create main dashboard HTML with 5 tabs.
        
        Args:
            timestamp (str): Timestamp string
            output_file (str): Output HTML file path
        """
        dashboard_content = f"""
        <!DOCTYPE html>
        <html lang='en'>
        <head>
            <meta charset='UTF-8'>
            <title>Crypto Trading Dashboard - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f8f8f8; margin: 0; padding: 20px; }}
                h1 {{ color: #222; text-align: center; }}
                .tabs {{ display: flex; border-bottom: 2px solid #ccc; margin-bottom: 20px; justify-content: center; flex-wrap: wrap; }}
                .tab {{ padding: 15px 30px; cursor: pointer; background: #eee; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 5px; margin-bottom: 5px; font-size: 14px; transition: all 0.3s; }}
                .tab:hover {{ background: #ddd; }}
                .tab.active {{ background: #fff; border-bottom: 2px solid #fff; font-weight: bold; color: #4CAF50; }}
                .tab-content {{ display: none; }}
                .tab-content.active {{ display: block; }}
                .iframe-box {{ background: #fff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                iframe {{ width: 100%; height: 700px; border: none; border-radius: 6px; }}
            </style>
            <script>
                function showTab(tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].classList.remove("active");
                    }}
                    tablinks = document.getElementsByClassName("tab");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].classList.remove("active");
                    }}
                    document.getElementById(tabName).classList.add("active");
                    document.getElementById(tabName + "-tab").classList.add("active");
                }}
                window.onload = function() {{ showTab('data_table'); }};
            </script>
        </head>
        <body>
            <h1>🚀 Crypto Trading Dashboard - Random Forest Model</h1>
            <p style="text-align: center; color: #666;">Generated: {timestamp}</p>
            
            <div class="tabs">
                <div class="tab active" id="data_table-tab" onclick="showTab('data_table')">📊 Full Data Table</div>
                <div class="tab" id="model_info-tab" onclick="showTab('model_info')">🤖 Model Information</div>
                <div class="tab" id="signals-tab" onclick="showTab('signals')">📈 Trading Signals</div>
                <div class="tab" id="train_data-tab" onclick="showTab('train_data')">🎓 Training Data Table</div>
                <div class="tab" id="test_data-tab" onclick="showTab('test_data')">🧪 Testing Data Table</div>
            </div>
            
            <div class="tab-content active" id="data_table">
                <div class="iframe-box">
                    <iframe src="data_table.html"></iframe>
                </div>
            </div>
            
            <div class="tab-content" id="model_info">
                <div class="iframe-box">
                    <iframe src="model_info.html"></iframe>
                </div>
            </div>
            
            <div class="tab-content" id="signals">
                <div class="iframe-box">
                    <iframe src="trading_signals.html"></iframe>
                </div>
            </div>
            
            <div class="tab-content" id="train_data">
                <div class="iframe-box">
                    <iframe src="training_data_table.html"></iframe>
                </div>
            </div>
            
            <div class="tab-content" id="test_data">
                <div class="iframe-box">
                    <iframe src="testing_data_table.html"></iframe>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(dashboard_content)
        print(f"\n✅ Dashboard generated successfully at {output_file}")
    
    def generate_full_report(self, reports_dir, timestamp, X_train, X_test, y_train, y_test, 
                            train_predictions, test_predictions, train_metrics, test_metrics, 
                            model_params, stop_loss_pct, target_pct):
        """
        Generate complete report with all visualizations and data tables in one call.
        
        Args:
            reports_dir (str): Directory to save reports
            timestamp (str): Timestamp string
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            train_predictions: Model predictions on training data
            test_predictions: Model predictions on test data
            train_metrics (dict): Training metrics
            test_metrics (dict): Test metrics
            model_params (dict): Model parameters
            stop_loss_pct (float): Stop loss percentage
            target_pct (float): Target percentage
        """
        import os
        
        print("\n📊 Generating comprehensive report...")
        print("-" * 60)
        
        # 1. Export full data table
        print("Creating full data table...")
        html_data_filename = os.path.join(reports_dir, "data_table.html")
        self.data.to_html(html_data_filename)
        
        # 2. Create trading signals plot
        print("Creating trading signals plot...")
        signals_filename = os.path.join(reports_dir, "trading_signals.html")
        self.create_trading_signals_plot(
            train_idx=X_train.index,
            test_idx=X_test.index,
            output_file=signals_filename
        )
        
        # 3. Create model information page
        print("Creating model information page...")
        model_info_filename = os.path.join(reports_dir, "model_info.html")
        self.create_model_info_html(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            model_params=model_params,
            num_train_samples=len(X_train),
            num_test_samples=len(X_test),
            num_train_target_1=sum(y_train),
            num_test_target_1=sum(y_test),
            num_train_signals=train_metrics['num_signals'],
            num_test_signals=test_metrics['num_signals'],
            cm=test_metrics['confusion_matrix'],
            class_report=test_metrics['classification_report'],
            stop_loss_pct=stop_loss_pct,
            target_pct=target_pct,
            output_file=model_info_filename
        )
        
        # 4. Create training data table
        print("Creating training data table with predictions...")
        training_data_filename = os.path.join(reports_dir, "training_data_table.html")
        self.create_training_data_table(
            X_train=X_train,
            y_train=y_train,
            predictions=train_predictions,
            data=self.data,
            output_file=training_data_filename
        )
        
        # 5. Create testing data table
        print("Creating testing data table with predictions...")
        testing_data_filename = os.path.join(reports_dir, "testing_data_table.html")
        self.create_testing_data_table(
            X_test=X_test,
            y_test=y_test,
            predictions=test_predictions,
            data=self.data,
            output_file=testing_data_filename
        )
        
        # 6. Create main dashboard
        print("Generating main dashboard...")
        dashboard_path = os.path.join(reports_dir, "dashboard.html")
        self.create_dashboard(
            timestamp=timestamp,
            output_file=dashboard_path
        )
        
        print("\n" + "="*60)
        print("✅ REPORT GENERATION COMPLETE!")
        print("="*60)
        print(f"\n📁 All reports saved to: {reports_dir}")
        print(f"🌐 Open dashboard: {dashboard_path}")
        print("\nDashboard includes 5 tabs:")
        print("  1. 📊 Full Data Table - Complete dataset with all features")
        print("  2. 🤖 Model Information - Model performance metrics")
        print("  3. 📈 Trading Signals - Interactive price chart with signals")
        print("  4. 🎓 Training Data Table - Training data with predictions")
        print("  5. 🧪 Testing Data Table - Testing data with predictions")
        print("\n" + "="*60 + "\n")
        
        # Open dashboard in Chrome automatically
        self._open_in_chrome(dashboard_path)
        
        return dashboard_path

    def _open_in_chrome(self, file_path):
        """
        Open an HTML file in Chrome browser.
        
        Args:
            file_path (str): Path to the HTML file to open
        """
        try:
            abs_path = os.path.abspath(file_path)
            
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", "-a", "Google Chrome", abs_path], check=True)
            elif platform.system() == "Windows":
                subprocess.run(["start", "chrome", abs_path], shell=True, check=True)
            elif platform.system() == "Linux":
                subprocess.run(["google-chrome", abs_path], check=True)
            
            print("🌐 Opening dashboard in Chrome...")
        except Exception as e:
            print(f"⚠️  Could not open Chrome: {e}")
            print(f"   Please open manually: {file_path}")
