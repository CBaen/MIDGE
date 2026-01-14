#!/usr/bin/env python3
"""
midge_server.py - Dashboard Server for Guiding Light

Serves the MIDGE dashboard with:
- Daily alerts API
- Prediction stats
- Plain language patterns

Run with: python dashboard/midge_server.py
Access at: http://localhost:8080
"""

import json
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Import MIDGE components
from dashboard.alerts import AlertGenerator
from trading.outcome_tracker import OutcomeTracker
from trading.learning_loop import LearningLoop


class MIDGEHandler(SimpleHTTPRequestHandler):
    """HTTP handler for MIDGE dashboard."""

    def __init__(self, *args, **kwargs):
        self.alert_generator = AlertGenerator()
        self.outcome_tracker = OutcomeTracker()
        self.learning_loop = LearningLoop()
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.serve_dashboard()
        elif parsed.path == "/api/alerts":
            self.send_json(self.get_alerts())
        elif parsed.path == "/api/stats":
            self.send_json(self.get_stats())
        elif parsed.path == "/api/predictions":
            self.send_json(self.get_predictions())
        elif parsed.path == "/api/reliability":
            self.send_json(self.get_reliability())
        else:
            super().do_GET()

    def send_json(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def serve_dashboard(self):
        """Serve the main dashboard HTML."""
        try:
            template_path = Path(__file__).parent / "templates" / "guiding_light.html"
            html = template_path.read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        except Exception as e:
            self.send_error(500, f"Error loading dashboard: {e}")

    def get_alerts(self):
        """Get current alerts with stats."""
        try:
            # Generate alerts
            alerts = self.alert_generator.generate_daily_alerts(
                min_confidence=0.4,
                max_alerts=15
            )

            # Get stats
            stats = self.outcome_tracker.get_overall_stats()

            return {
                "alerts": [
                    {
                        "level": a.level,
                        "symbol": a.symbol,
                        "headline": a.headline,
                        "details": a.details,
                        "confidence": a.confidence,
                        "source": a.source,
                        "action_suggestion": a.action_suggestion,
                        "trade_value": a.trade_value,
                        "contract_value": a.contract_value,
                        "trader_name": a.trader_name
                    }
                    for a in alerts
                ],
                "stats": stats,
                "generated_at": self.alert_generator._last_generated.isoformat()
                    if self.alert_generator._last_generated else None
            }
        except Exception as e:
            print(f"Error generating alerts: {e}")
            return {"alerts": [], "stats": {}, "error": str(e)}

    def get_stats(self):
        """Get overall system stats."""
        return {
            "predictions": self.outcome_tracker.get_overall_stats(),
            "by_timeframe": self.outcome_tracker.get_performance_by_timeframe(),
            "by_direction": self.outcome_tracker.get_performance_by_direction()
        }

    def get_predictions(self):
        """Get pending predictions."""
        pending = self.outcome_tracker.get_pending_predictions()
        return {
            "pending": [
                {
                    "id": p.prediction_id,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "confidence": p.confidence,
                    "entry_price": p.entry_price,
                    "predicted_at": p.predicted_at,
                    "outcome_due": p.outcome_due
                }
                for p in pending[:20]
            ],
            "total": len(pending)
        }

    def get_reliability(self):
        """Get signal reliability scores."""
        return {
            "scores": self.learning_loop.get_all_reliabilities(),
            "signal_performance": self.outcome_tracker.get_signal_performance()
        }


def run_server(port=8080):
    """Start the MIDGE dashboard server."""
    os.chdir(Path(__file__).parent)
    server = HTTPServer(("localhost", port), MIDGEHandler)
    print(f"MIDGE Dashboard running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
