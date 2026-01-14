#!/usr/bin/env python3
"""
alerts.py - Daily Alert Generator for Guiding Light

Plain language. No trader jargon. Just patterns and confidence.

"Here's what I found. Here's why it matters. You decide."
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Import signal generators
from trading.edge.politician_tracker import PoliticianTracker, get_daily_alerts as get_politician_alerts
from trading.technical.signals import SignalGenerator


@dataclass
class Alert:
    """A single alert for Guiding Light."""
    level: str  # "STRONG", "MEDIUM", "WATCH"
    symbol: str
    headline: str  # One-line summary
    details: str  # 2-3 sentence explanation
    confidence: float
    source: str  # "politician", "insider", "technical", "contract"
    timestamp: str = ""

    # Optional enrichment
    trade_value: float = 0.0
    contract_value: float = 0.0
    trader_name: str = ""
    action_suggestion: str = ""  # Informational only

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class AlertGenerator:
    """
    Generates daily alerts from all signal sources.

    Priority order:
    1. Politician/contract correlations (highest edge)
    2. Insider trades (Form 4)
    3. Technical signals (confirmation)
    """

    def __init__(self):
        self.politician_tracker = PoliticianTracker()
        self._alerts_cache: List[Alert] = []
        self._last_generated: Optional[datetime] = None

    def generate_daily_alerts(self,
                             symbols: List[str] = None,
                             min_confidence: float = 0.5,
                             max_alerts: int = 10) -> List[Alert]:
        """
        Generate today's alerts from all sources.

        Args:
            symbols: Specific symbols to check (None = scan defaults)
            min_confidence: Minimum confidence threshold
            max_alerts: Maximum alerts to return

        Returns:
            List of Alert objects, sorted by confidence
        """
        alerts = []

        # 1. Politician/contract correlations (primary edge)
        politician_alerts = self._get_politician_alerts(symbols, min_confidence)
        alerts.extend(politician_alerts)

        # 2. Could add more sources here:
        # - Institutional 13F alerts
        # - Unusual options activity
        # - Technical breakout alerts

        # Sort by confidence and limit
        alerts.sort(key=lambda x: x.confidence, reverse=True)
        self._alerts_cache = alerts[:max_alerts]
        self._last_generated = datetime.now()

        return self._alerts_cache

    def _get_politician_alerts(self,
                               symbols: List[str] = None,
                               min_confidence: float = 0.5) -> List[Alert]:
        """Convert politician correlations to alerts."""
        try:
            correlations = self.politician_tracker.find_correlations(
                symbols=symbols,
                days_lookback=90,
                min_trade_value=25000
            )
        except Exception as e:
            print(f"Politician tracker error: {e}")
            return []

        alerts = []
        for corr in correlations:
            if corr.confidence < min_confidence:
                continue

            # Determine level
            if corr.confidence >= 0.8:
                level = "STRONG"
            elif corr.confidence >= 0.65:
                level = "MEDIUM"
            else:
                level = "WATCH"

            # Create headline based on correlation type
            if corr.correlation_type == "politician_contract":
                if corr.oversight_match:
                    headline = f"{corr.trader_name} bought {corr.symbol} before agency they oversee awarded contract"
                else:
                    headline = f"{corr.trader_name} bought {corr.symbol} before related contract awarded"
            elif corr.correlation_type == "insider_preannouncement":
                headline = f"Insider {corr.trader_name} made significant {corr.symbol} purchase"
            else:
                headline = f"{corr.trader_name} traded {corr.symbol}"

            # Build details
            details = corr.to_plain_language()
            if corr.oversight_match:
                details += f" This politician sits on a committee that oversees the awarding agency."

            # Action suggestion (informational only)
            if corr.confidence >= 0.8:
                action = "Strong pattern. Worth researching further."
            elif corr.confidence >= 0.65:
                action = "Moderate pattern. Monitor for confirmation."
            else:
                action = "Weak pattern. Watch but don't act on this alone."

            alerts.append(Alert(
                level=level,
                symbol=corr.symbol,
                headline=headline,
                details=details,
                confidence=corr.confidence,
                source="politician" if corr.correlation_type == "politician_contract" else "insider",
                trade_value=corr.value,
                contract_value=corr.contract_value if corr.contract else 0,
                trader_name=corr.trader_name,
                action_suggestion=action
            ))

        return alerts

    def format_for_dashboard(self) -> str:
        """
        Format alerts for display in Guiding Light's dashboard.

        Simple. Readable. No jargon.
        """
        if not self._alerts_cache:
            self.generate_daily_alerts()

        output = []
        output.append("=" * 60)
        output.append(f"TODAY'S PATTERNS - {datetime.now().strftime('%B %d, %Y')}")
        output.append("=" * 60)
        output.append("")

        if not self._alerts_cache:
            output.append("No significant patterns detected today.")
            output.append("")
            output.append("This could mean:")
            output.append("  - Markets are quiet")
            output.append("  - No insider activity in tracked symbols")
            output.append("  - Existing patterns don't meet confidence threshold")
            output.append("")
            output.append("Check back tomorrow. Patience is part of the edge.")
            return "\n".join(output)

        # Group by level
        strong = [a for a in self._alerts_cache if a.level == "STRONG"]
        medium = [a for a in self._alerts_cache if a.level == "MEDIUM"]
        watch = [a for a in self._alerts_cache if a.level == "WATCH"]

        for group, label in [(strong, "STRONG SIGNALS"),
                            (medium, "MODERATE SIGNALS"),
                            (watch, "WATCHING")]:
            if group:
                output.append(f"--- {label} ---")
                output.append("")

                for alert in group:
                    output.append(f"[{alert.level}] {alert.symbol}")
                    output.append(f"  {alert.headline}")
                    output.append(f"  {alert.details}")
                    output.append(f"  Confidence: {alert.confidence:.0%}")
                    output.append(f"  {alert.action_suggestion}")
                    output.append("")

        output.append("=" * 60)
        output.append("Remember: This is information, not advice.")
        output.append("You decide what to do with it.")
        output.append("=" * 60)

        return "\n".join(output)

    def to_json(self) -> str:
        """Export alerts as JSON for API consumption."""
        if not self._alerts_cache:
            self.generate_daily_alerts()

        return json.dumps({
            "generated_at": self._last_generated.isoformat() if self._last_generated else None,
            "alert_count": len(self._alerts_cache),
            "alerts": [asdict(a) for a in self._alerts_cache]
        }, indent=2)

    def save_daily_report(self, output_dir: str = None) -> str:
        """Save daily report to file."""
        if output_dir is None:
            output_dir = Path("C:/Users/baenb/projects/MIDGE/reports")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = output_dir / f"daily_alerts_{date_str}.txt"
        json_path = output_dir / f"daily_alerts_{date_str}.json"

        # Save both formats
        report_path.write_text(self.format_for_dashboard())
        json_path.write_text(self.to_json())

        return str(report_path)


def generate_alerts(symbols: List[str] = None,
                   min_confidence: float = 0.5) -> List[Alert]:
    """Convenience function to generate alerts."""
    generator = AlertGenerator()
    return generator.generate_daily_alerts(symbols, min_confidence)


def print_daily_report():
    """Print today's report to console."""
    generator = AlertGenerator()
    print(generator.format_for_dashboard())


if __name__ == "__main__":
    print("MIDGE Alert Generator")
    print()

    generator = AlertGenerator()

    # Generate alerts for key symbols
    alerts = generator.generate_daily_alerts(
        symbols=["LMT", "RTX", "BA", "MSFT", "AAPL"],
        min_confidence=0.4
    )

    print(f"Generated {len(alerts)} alerts")
    print()

    # Print formatted report
    print(generator.format_for_dashboard())

    # Save report
    try:
        path = generator.save_daily_report()
        print(f"\nReport saved to: {path}")
    except Exception as e:
        print(f"\nCould not save report: {e}")
