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
    
    def create_trading_signals_plot(self, train_idx, test_idx, output_file, backtest_trades=None, test_predictions=None):
        """
        Create trading signals plot with color-coded signals:
        - Blue: Training signals (target=1)
        - Green: Test signals that HIT target profit
        - Red: Test signals that hit stop loss
        - Yellow: Test signals still open at period end
        
        Args:
            train_idx: Training data indices
            test_idx: Test data indices
            output_file (str): Output HTML file path
            backtest_trades (list): List of trades from backtest to determine signal outcome
            test_predictions (list): Predicted signals for test data (used to filter chart signals)
        """
        df = self.data.copy()
        
        # Extract entry dates from backtest trades by exit reason
        target_hit_dates = set()
        stop_loss_dates = set()
        period_end_dates = set()
        
        if backtest_trades:
            for trade in backtest_trades:
                entry_date = trade.get('entry_date', '')
                exit_reason = trade.get('exit_reason', '')
                
                if exit_reason == 'Target Hit':
                    target_hit_dates.add(entry_date)
                elif exit_reason == 'Stop Loss':
                    stop_loss_dates.add(entry_date)
                elif exit_reason == 'Period End':
                    period_end_dates.add(entry_date)
        
        fig = go.Figure()
        
        # Plot price line
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Close'], 
            mode='lines', 
            name='Close Price', 
            line=dict(color='black', width=1)
        ))
        
        # Plot training signals with target=1 (BLUE dots)
        train_target_1_mask = (df.index.isin(train_idx)) & (df['signal_labels'] == 1)
        fig.add_trace(go.Scatter(
            x=df.index[train_target_1_mask],
            y=df['Close'][train_target_1_mask],
            mode='markers',
            marker=dict(color='blue', size=8, symbol='circle'),
            name='Blue: Training Signals'
        ))
        
        # Plot test signals based on PREDICTED signals (not actual signal_labels)
        # Filter test data to only predicted signals (prediction == 1)
        if test_predictions is not None:
            test_indices_list = list(test_idx)
            test_signals_mask = (df.index.isin(test_idx)) & (df.index.isin([test_indices_list[i] for i, pred in enumerate(test_predictions) if pred == 1]))
        else:
            # Fallback to actual signals if predictions not provided
            test_signals_mask = (df.index.isin(test_idx)) & (df['signal_labels'] == 1)
        
        test_signals_df = df[test_signals_mask]
        
        # Plot test signals that HIT target (GREEN dots)
        if backtest_trades and target_hit_dates:
            hit_mask = test_signals_df.index.astype(str).isin(target_hit_dates)
            fig.add_trace(go.Scatter(
                x=test_signals_df.index[hit_mask],
                y=test_signals_df['Close'][hit_mask],
                mode='markers',
                marker=dict(color='green', size=10, symbol='circle', line=dict(color='darkgreen', width=2)),
                name='Green: Target Hit'
            ))
        
        # Plot test signals that HIT STOP LOSS (RED dots)
        if backtest_trades and stop_loss_dates:
            miss_mask = test_signals_df.index.astype(str).isin(stop_loss_dates)
            fig.add_trace(go.Scatter(
                x=test_signals_df.index[miss_mask],
                y=test_signals_df['Close'][miss_mask],
                mode='markers',
                marker=dict(color='red', size=10, symbol='circle', line=dict(color='darkred', width=2)),
                name='Red: Stop Loss Hit'
            ))
        
        # Plot test signals STILL OPEN at period end (YELLOW dots)
        if backtest_trades and period_end_dates:
            period_mask = test_signals_df.index.astype(str).isin(period_end_dates)
            fig.add_trace(go.Scatter(
                x=test_signals_df.index[period_mask],
                y=test_signals_df['Close'][period_mask],
                mode='markers',
                marker=dict(color='gold', size=10, symbol='circle', line=dict(color='orange', width=2)),
                name='Yellow: Still Open at Period End'
            ))
        
        # If no backtest data, show all test signals as default (for compatibility)
        elif not backtest_trades:
            fig.add_trace(go.Scatter(
                x=test_signals_df.index,
                y=test_signals_df['Close'],
                mode='markers',
                marker=dict(color='blue', size=8, symbol='circle'),
                name='Test Signals'
            ))
        
        fig.update_layout(
            title=f"Random Forest Trading Signals - {self.symbol}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=700,
            showlegend=False  # Disable Plotly legend, we'll add custom HTML legend
        )
        
        # Create custom HTML with legend above the chart
        chart_html = fig.to_html(include_plotlyjs='cdn')
        
        legend_html = """
        <!DOCTYPE html>
        <html lang='en'>
        <head>
            <meta charset='UTF-8'>
            <title>Trading Signals Chart</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #f8f8f8;
                    padding: 20px;
                    margin: 0;
                }
                .legend-container {
                    background: white;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    border: 2px solid #333;
                }
                .legend-title {
                    font-size: 16px;
                    font-weight: bold;
                    color: #222;
                    margin-bottom: 12px;
                    text-align: center;
                }
                .legend-items {
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr 1fr;
                    gap: 12px;
                }
                .legend-item {
                    display: flex;
                    align-items: center;
                    padding: 10px;
                    background: #f0f0f0;
                    border-radius: 6px;
                    font-size: 13px;
                }
                .legend-color {
                    width: 22px;
                    height: 22px;
                    border-radius: 50%;
                    margin-right: 10px;
                    flex-shrink: 0;
                    border: 2px solid #333;
                }
                .legend-text {
                    font-weight: bold;
                    color: #222;
                }
                .blue { background: #0066CC; }
                .green { background: #00AA00; border: 2px solid #006600 !important; }
                .red { background: #CC0000; border: 2px solid #660000 !important; }
                .yellow { background: #FFD700; border: 2px solid #FF8C00 !important; }
                .chart-container {
                    background: white;
                    padding: 10px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
            </style>
        </head>
        <body>
            <div class="legend-container">
                <div class="legend-title">📊 Signal Legend</div>
                <div class="legend-items">
                    <div class="legend-item">
                        <div class="legend-color blue"></div>
                        <div class="legend-text">Blue: Training Signals</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color green"></div>
                        <div class="legend-text">Green: Target Hit</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color red"></div>
                        <div class="legend-text">Red: Stop Loss Hit</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color yellow"></div>
                        <div class="legend-text">Yellow: Still Open</div>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
        """
        
        # Insert the legend before the chart
        full_html = legend_html + chart_html + """
            </div>
        </body>
        </html>
        """
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"Exported trading signals plot to {output_file}")
    
    def create_model_info_html(self, train_metrics, test_metrics, model_params, 
                               num_train_samples, num_test_samples, 
                               num_train_target_1, num_test_target_1,
                               num_train_signals, num_test_signals,
                               cm, class_report, stop_loss_pct, target_pct,
                               output_file, feature_importance=None, feature_names=None):
        """
        Create model information HTML page with Trading Parameters on top.
        """
        # Build feature importance table if available
        feature_importance_html = ""
        if feature_importance is not None and feature_names is not None:
            # Sort features by importance (descending)
            sorted_indices = sorted(range(len(feature_importance)), key=lambda i: feature_importance[i], reverse=True)
            
            feature_importance_html = """
            <div class="info-section">
                <h2>Feature Importance (All Features)</h2>
                <table>
                    <tr><th>Rank</th><th>Feature</th><th>Importance Weight</th><th>Percentage</th></tr>
            """
            for rank, idx in enumerate(sorted_indices, 1):
                feature_name = feature_names[idx]
                importance = feature_importance[idx]
                percentage = importance * 100
                feature_importance_html += f"""
                    <tr><td>{rank}</td><td>{feature_name}</td><td>{importance:.6f}</td><td>{percentage:.2f}%</td></tr>
                """
            feature_importance_html += """
                </table>
            </div>
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
            
            {feature_importance_html}
            
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
        </body>
        </html>
        """
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(model_info_html)
        print(f"Exported model information to {output_file}")
    
    def create_training_data_table(self, X_train, y_train, predictions, output_file):
        """
        Create training data table with expected and predicted signals.
        """
        train_data = X_train.copy()
        train_data['expected_signal_labels'] = y_train
        train_data['predicted_signal_labels'] = predictions
        
        train_data.to_html(output_file)
        print(f"Exported training data table to {output_file}")
    
    def create_testing_data_table(self, X_test, y_test, predictions, output_file):
        """
        Create testing data table with expected and predicted signals.
        """
        test_data = X_test.copy()
        test_data['expected_signal_labels'] = y_test
        test_data['predicted_signal_labels'] = predictions
        
        test_data.to_html(output_file)
        print(f"Exported testing data table to {output_file}")
    
    def create_backtest_results_html(self, backtest_data, output_file):
        """
        Create backtest results HTML page with trading statistics.
        """
        trades = backtest_data.get('trades', [])
        total_return = backtest_data.get('total_return', 0)
        win_rate = backtest_data.get('win_rate', 0)
        avg_win = backtest_data.get('avg_win', 0)
        avg_loss = backtest_data.get('avg_loss', 0)
        num_trades = backtest_data.get('num_trades', 0)
        num_wins = backtest_data.get('num_wins', 0)
        num_losses = backtest_data.get('num_losses', 0)
        final_balance = backtest_data.get('final_balance', 10000)
        initial_balance = backtest_data.get('initial_balance', 10000)
        
        # Sort trades by entry date
        sorted_trades = sorted(trades, key=lambda x: str(x.get('entry_date', '')))
        
        trades_html = ""
        if sorted_trades:
            for i, trade in enumerate(sorted_trades, 1):
                entry_date = str(trade.get('entry_date', 'N/A')).split()[0]
                exit_date = str(trade.get('exit_date', 'N/A')).split()[0]
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                pnl = trade.get('pnl', 0)
                pnl_pct = trade.get('pnl_pct', 0)
                exit_reason = trade.get('exit_reason', 'N/A')
                number_of_shares = trade.get('number_of_shares', 0)
                balance = trade.get('balance', 0)
                color = '#28a745' if pnl >= 0 else '#dc3545'
                trades_html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{entry_date}</td>
                    <td>{exit_date}</td>
                    <td>${entry_price:.2f}</td>
                    <td>${exit_price:.2f}</td>
                    <td>{number_of_shares:.4f}</td>
                    <td style="color: {color}; font-weight: bold;">${pnl:.2f}</td>
                    <td style="color: {color}; font-weight: bold;">{pnl_pct:.2f}%</td>
                    <td>${balance:,.2f}</td>
                    <td>{exit_reason}</td>
                </tr>
                """
        
        backtest_html = f"""
        <!DOCTYPE html>
        <html lang='en'>
        <head>
            <meta charset='UTF-8'>
            <title>Backtest Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f8f8f8; padding: 20px; }}
                h1, h2 {{ color: #222; }}
                .results-section {{ background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                .summary-box {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 10px 0; }}
                .metric-box {{ flex: 1; min-width: 200px; padding: 15px; background: #f0f0f0; border-radius: 6px; text-align: center; }}
                .metric-label {{ color: #666; font-weight: bold; margin-bottom: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>💰 Backtest Results - Test Data Period</h1>
            <p style="color: #666;">Analysis based on ${initial_balance} initial capital</p>
            
            <div class="results-section">
                <h2>Trading Summary</h2>
                <div class="summary-box">
                    <div class="metric-box">
                        <div class="metric-label">Total Trades</div>
                        <div class="metric-value">{num_trades}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Winning Trades</div>
                        <div class="metric-value" style="color: #28a745;">{num_wins}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Losing Trades</div>
                        <div class="metric-value" style="color: #dc3545;">{num_losses}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{win_rate:.2f}%</div>
                    </div>
                </div>
            </div>
            
            <div class="results-section">
                <h2>Financial Performance</h2>
                <div class="summary-box">
                    <div class="metric-box">
                        <div class="metric-label">Initial Balance</div>
                        <div class="metric-value">${initial_balance:,.2f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Final Balance</div>
                        <div class="metric-value" style="color: {'#28a745' if final_balance >= initial_balance else '#dc3545'};">${final_balance:,.2f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Total P&L</div>
                        <div class="metric-value" style="color: {'#28a745' if total_return >= 0 else '#dc3545'};">${total_return:,.2f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Return %</div>
                        <div class="metric-value" style="color: {'#28a745' if (total_return/initial_balance)*100 >= 0 else '#dc3545'};">{(total_return/initial_balance)*100:.2f}%</div>
                    </div>
                </div>
            </div>
            
            <div class="results-section">
                <h2>Trade Statistics</h2>
                <div class="summary-box">
                    <div class="metric-box">
                        <div class="metric-label">Average Win</div>
                        <div class="metric-value" style="color: #28a745;">${avg_win:.2f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Average Loss</div>
                        <div class="metric-value" style="color: #dc3545;">${avg_loss:.2f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value">{'N/A' if num_losses == 0 else f"{abs(avg_win * num_wins) / abs(avg_loss * num_losses):.2f}"}</div>
                    </div>
                </div>
            </div>
            
            <div class="results-section">
                <h2>Individual Trades</h2>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Entry Date</th>
                        <th>Exit Date</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Shares</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Portfolio Balance</th>
                        <th>Exit Reason</th>
                    </tr>
                    {trades_html if trades_html else '<tr><td colspan="10" style="text-align: center;">No trades executed</td></tr>'}
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(backtest_html)
        print(f"Exported backtest results to {output_file}")
    
    def create_dashboard(self, timestamp, output_file, symbol, stop_loss_pct, target_pct, 
                        initial_balance=10000, each_trade_value=1000):
        """
        Create main dashboard HTML with 6 tabs and trading parameters on main page.
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
                .params-box {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px; text-align: center; }}
                .param-item {{ display: inline-block; margin: 0 30px; }}
                .param-label {{ font-weight: bold; color: #666; }}
                .param-value {{ font-size: 18px; color: #4CAF50; font-weight: bold; }}
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
            
            <div class="params-box">
                <div class="param-item">
                    <div class="param-label">Symbol</div>
                    <div class="param-value">{symbol}</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Stop Loss</div>
                    <div class="param-value">{stop_loss_pct}%</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Target Profit</div>
                    <div class="param-value">{target_pct}%</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Trade Value</div>
                    <div class="param-value">${each_trade_value}</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Initial Balance</div>
                    <div class="param-value">${initial_balance:,.0f}</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Data Source</div>
                    <div class="param-value">Yahoo Finance</div>
                </div>
            </div>
            
            <div class="tabs">
                <div class="tab active" id="data_table-tab" onclick="showTab('data_table')">📊 Full Data Table</div>
                <div class="tab" id="train_data-tab" onclick="showTab('train_data')">🎓 Training Data Table</div>
                <div class="tab" id="test_data-tab" onclick="showTab('test_data')">🧪 Testing Data Table</div>
                <div class="tab" id="model_info-tab" onclick="showTab('model_info')">🤖 Model Information</div>
                <div class="tab" id="signals-tab" onclick="showTab('signals')">📈 Trading Signals</div>
                <div class="tab" id="backtest-tab" onclick="showTab('backtest')">💰 Backtest Results</div>
            </div>
            
            <div class="tab-content active" id="data_table">
                <div class="iframe-box">
                    <iframe src="data_table.html"></iframe>
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
            
            <div class="tab-content" id="backtest">
                <div class="iframe-box">
                    <iframe src="backtest_results.html"></iframe>
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
                            model_params, stop_loss_pct, target_pct, initial_balance=10000, 
                            each_trade_value=1000, model=None):
        """
        Generate complete report with all visualizations and data tables in new tab order.
        """
        import os
        
        print("\n📊 Generating comprehensive report...")
        print("-" * 60)
        
        # 1. Export full data table
        print("Creating full data table...")
        html_data_filename = os.path.join(reports_dir, "data_table.html")
        self.data.to_html(html_data_filename)
        
        # 2. Create training data table
        print("Creating training data table with predictions...")
        training_data_filename = os.path.join(reports_dir, "training_data_table.html")
        self.create_training_data_table(
            X_train=X_train,
            y_train=y_train,
            predictions=train_predictions,
            output_file=training_data_filename
        )
        
        # 3. Create testing data table
        print("Creating testing data table with predictions...")
        testing_data_filename = os.path.join(reports_dir, "testing_data_table.html")
        self.create_testing_data_table(
            X_test=X_test,
            y_test=y_test,
            predictions=test_predictions,
            output_file=testing_data_filename
        )
        
        # 4. Create model information page
        print("Creating model information page...")
        model_info_filename = os.path.join(reports_dir, "model_info.html")
        
        # Extract feature importance if model is provided
        feature_importance = None
        if model is not None and hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            feature_importance = model.model.feature_importances_
        
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
            output_file=model_info_filename,
            feature_importance=feature_importance,
            feature_names=list(X_train.columns) if hasattr(X_train, 'columns') else None
        )
        
        # 5. Create backtest results FIRST (before signals plot so we can use trade data)
        print("Creating backtest analysis based on predicted signals...")
        backtest_filename = os.path.join(reports_dir, "backtest_results.html")
        backtest_data = self._run_backtest(X_test, test_predictions, self.data.loc[X_test.index], 
                                         stop_loss_pct, target_pct, initial_balance, each_trade_value)
        self.create_backtest_results_html(
            backtest_data=backtest_data,
            output_file=backtest_filename
        )
        
        # 6. Create trading signals plot WITH backtest data to color signals
        print("Creating trading signals plot...")
        signals_filename = os.path.join(reports_dir, "trading_signals.html")
        self.create_trading_signals_plot(
            train_idx=X_train.index,
            test_idx=X_test.index,
            output_file=signals_filename,
            backtest_trades=backtest_data.get('trades', []),
            test_predictions=test_predictions
        )
        
        # 7. Create main dashboard
        print("Generating main dashboard...")
        dashboard_path = os.path.join(reports_dir, "dashboard.html")
        self.create_dashboard(
            timestamp=timestamp,
            output_file=dashboard_path,
            symbol=self.symbol,
            stop_loss_pct=stop_loss_pct,
            target_pct=target_pct,
            initial_balance=initial_balance,
            each_trade_value=each_trade_value
        )
        
        print("\n" + "="*60)
        print("✅ REPORT GENERATION COMPLETE!")
        print("="*60)
        print(f"\n📁 All reports saved to: {reports_dir}")
        print(f"🌐 Open dashboard: {dashboard_path}")
        print("\nDashboard includes 6 tabs:")
        print("  1. 📊 Full Data Table - Complete dataset with all features")
        print("  2. 🎓 Training Data Table - Training data with predictions")
        print("  3. 🧪 Testing Data Table - Testing data with predictions")
        print("  4. 🤖 Model Information - Model performance metrics (Trading Parameters on top)")
        print("  5. 📈 Trading Signals - Interactive price chart with signals")
        print("  6. 💰 Backtest Results - Backtest performance with $10,000 capital")
        print("\n" + "="*60 + "\n")
        
        # Open dashboard in Chrome automatically
        self._open_in_chrome(dashboard_path)
        
        return dashboard_path
    
    def _run_backtest(self, X_test, test_predictions, test_data, stop_loss_pct, target_pct, 
                     initial_balance=10000, each_trade_value=1000):
        """
        Run backtest on test data PREDICTED signals with multiple simultaneous positions.
        Each position uses each_trade_value USD, regardless of balance.
        
        Args:
            X_test: Test feature indices
            test_predictions: Predicted signals (1 for buy, 0 for no signal)
            test_data: OHLCV data for test period
            stop_loss_pct: Stop loss percentage
            target_pct: Target profit percentage
            initial_balance: Starting capital
            each_trade_value: Amount to risk per trade
        """
        trades = []
        balance = initial_balance
        positions = []  # List of open positions
        num_wins = 0
        num_losses = 0
        total_pnl = 0
        position_id_counter = 0
        
        # Get test data sorted by index
        test_data = test_data.sort_index()
        
        for idx_pos, (idx, row) in enumerate(test_data.iterrows()):
            if idx not in X_test.index:
                continue
            
            # Get the predicted signal (1 for buy, 0 for no signal)
            try:
                signal = int(test_predictions[idx_pos])
            except:
                signal = 0
            
            close_price = row.get('Close', 0)
            high_price = row.get('High', close_price)
            low_price = row.get('Low', close_price)
            
            # Entry logic: open a new position if we have a predicted buy signal
            if signal == 1:
                number_of_shares = each_trade_value / close_price
                position_id_counter += 1
                new_position = {
                    'position_id': position_id_counter,
                    'entry_date': idx,
                    'entry_price': close_price,
                    'entry_time': str(idx),
                    'number_of_shares': number_of_shares,
                    'target_price': close_price * (1 + target_pct / 100),
                    'stop_price': close_price * (1 - stop_loss_pct / 100),
                }
                positions.append(new_position)
            
            # Check exit conditions for all open positions
            positions_to_remove = []
            for i, position in enumerate(positions):
                # Check stop loss using LOW price
                if low_price <= position['stop_price']:
                    exit_price = position['stop_price']
                    pnl = (exit_price - position['entry_price']) * position['number_of_shares']
                    pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                    balance += pnl
                    trades.append({
                        'entry_date': position['entry_time'],
                        'exit_date': str(idx),
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'Stop Loss',
                        'number_of_shares': position['number_of_shares'],
                        'balance': balance
                    })
                    total_pnl += pnl
                    if pnl >= 0:
                        num_wins += 1
                    else:
                        num_losses += 1
                    positions_to_remove.append(i)
                
                # Check target profit using HIGH price (only if stop loss not hit)
                elif high_price >= position['target_price']:
                    exit_price = position['target_price']
                    pnl = (exit_price - position['entry_price']) * position['number_of_shares']
                    pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                    balance += pnl
                    trades.append({
                        'entry_date': position['entry_time'],
                        'exit_date': str(idx),
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'Target Hit',
                        'number_of_shares': position['number_of_shares'],
                        'balance': balance
                    })
                    total_pnl += pnl
                    num_wins += 1
                    positions_to_remove.append(i)
            
            # Remove closed positions (in reverse order to maintain indices)
            for i in reversed(positions_to_remove):
                positions.pop(i)
        
        # Close any remaining positions at the last price
        if positions:
            last_price = test_data.iloc[-1]['Close']
            for position in positions:
                pnl = (last_price - position['entry_price']) * position['number_of_shares']
                pnl_pct = ((last_price - position['entry_price']) / position['entry_price']) * 100
                balance += pnl
                trades.append({
                    'entry_date': position['entry_time'],
                    'exit_date': str(test_data.index[-1]),
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'Period End',
                    'number_of_shares': position['number_of_shares'],
                    'balance': balance
                })
                total_pnl += pnl
                if pnl >= 0:
                    num_wins += 1
                else:
                    num_losses += 1
        
        win_rate = (num_wins / len(trades) * 100) if trades else 0
        avg_win = sum(t['pnl'] for t in trades if t['pnl'] >= 0) / num_wins if num_wins > 0 else 0
        avg_loss = sum(t['pnl'] for t in trades if t['pnl'] < 0) / num_losses if num_losses > 0 else 0
        
        return {
            'trades': trades,
            'total_return': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'num_trades': len(trades),
            'num_wins': num_wins,
            'num_losses': num_losses,
            'final_balance': balance,
            'initial_balance': initial_balance
        }

    def _open_in_chrome(self, file_path):
        """
        Open an HTML file in Chrome browser.
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
