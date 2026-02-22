#!/usr/bin/env python3
"""
filing_time_analyzer.py - Detect patterns in SEC filing times

Filing time patterns often reveal intent:
- Friday 4:30 PM = hiding bad news (dump before weekend)
- After hours = avoiding immediate market reaction
- Monday 9 AM = confident/routine disclosure
- Pre-market = urgent, wants immediate reaction

This analyzer adds confidence modifiers to insider signals based on WHEN they filed.
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Optional, Dict
import requests


@dataclass
class FilingTimeSignal:
    """Signal derived from filing time patterns."""
    # Filing details
    ticker: str
    filer_name: str
    filing_date: str
    filing_time: str  # HH:MM format
    form_type: str  # "4" or "8-K"

    # Pattern detection
    pattern_type: str  # "hiding", "urgent", "routine", "suspicious"
    day_of_week: str  # "Monday", "Friday", etc.
    market_session: str  # "premarket", "market_hours", "after_hours"

    # Signal adjustment
    confidence_modifier: float  # -0.15 to +0.10
    explanation: str

    # Metadata
    signal_source: str = "filing_time"
    decay_rate: float = 0.10  # Fast decay - timing context is ephemeral

    def to_plain_language(self) -> str:
        """Format for dashboard."""
        mod = "+" if self.confidence_modifier >= 0 else ""
        return (
            f"{self.ticker} {self.form_type} filed {self.day_of_week} {self.filing_time}: "
            f"{self.pattern_type} ({mod}{self.confidence_modifier:.0%}) - {self.explanation}"
        )


class FilingTimeAnalyzer:
    """
    Analyzes filing timestamps for behavioral signals.

    Patterns detected:
    - Friday afternoon (3-5 PM): Often used to hide bad news
    - After hours (>4 PM): Avoiding immediate reaction
    - Pre-market (<9:30 AM): Urgent disclosure
    - Monday morning: Routine, confident disclosure
    """

    # Market hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    # Pattern definitions with confidence modifiers
    PATTERNS = {
        "friday_afternoon": {
            "modifier": -0.15,
            "pattern_type": "hiding",
            "explanation": "Friday afternoon filing - classic 'take out the trash' timing"
        },
        "friday_after_hours": {
            "modifier": -0.12,
            "pattern_type": "hiding",
            "explanation": "Friday after hours - maximizing time before market reaction"
        },
        "after_hours": {
            "modifier": -0.08,
            "pattern_type": "suspicious",
            "explanation": "After hours filing - avoiding immediate market reaction"
        },
        "premarket_urgent": {
            "modifier": +0.05,
            "pattern_type": "urgent",
            "explanation": "Pre-market filing - wants immediate market attention"
        },
        "monday_morning": {
            "modifier": 0.0,
            "pattern_type": "routine",
            "explanation": "Monday morning - routine, confident disclosure"
        },
        "market_hours": {
            "modifier": 0.0,
            "pattern_type": "routine",
            "explanation": "Standard market hours filing"
        }
    }

    def __init__(self):
        self.qdrant_url = "http://localhost:6333"

    def analyze_filing_time(
        self,
        ticker: str,
        filer_name: str,
        filing_date: str,
        filing_datetime: datetime,
        form_type: str = "4"
    ) -> FilingTimeSignal:
        """
        Analyze a single filing's timestamp for patterns.

        Args:
            ticker: Stock ticker
            filer_name: Name of the filer
            filing_date: Date string (YYYY-MM-DD)
            filing_datetime: Full datetime of filing
            form_type: "4" or "8-K"

        Returns:
            FilingTimeSignal with pattern analysis
        """
        filing_time = filing_datetime.time()
        day_of_week = filing_datetime.strftime("%A")
        time_str = filing_datetime.strftime("%H:%M")

        # Determine market session
        if filing_time < self.MARKET_OPEN:
            market_session = "premarket"
        elif filing_time > self.MARKET_CLOSE:
            market_session = "after_hours"
        else:
            market_session = "market_hours"

        # Detect pattern
        pattern_key = self._detect_pattern(day_of_week, filing_time, market_session)
        pattern_info = self.PATTERNS.get(pattern_key, self.PATTERNS["market_hours"])

        return FilingTimeSignal(
            ticker=ticker,
            filer_name=filer_name,
            filing_date=filing_date,
            filing_time=time_str,
            form_type=form_type,
            pattern_type=pattern_info["pattern_type"],
            day_of_week=day_of_week,
            market_session=market_session,
            confidence_modifier=pattern_info["modifier"],
            explanation=pattern_info["explanation"]
        )

    def _detect_pattern(self, day: str, filing_time: time, session: str) -> str:
        """Determine the pattern based on day and time."""
        # Friday patterns (highest suspicion)
        if day == "Friday":
            if filing_time >= time(15, 0) and filing_time <= time(17, 0):
                return "friday_afternoon"
            elif session == "after_hours":
                return "friday_after_hours"

        # General after-hours (not Friday)
        if session == "after_hours":
            return "after_hours"

        # Pre-market (any day)
        if session == "premarket":
            return "premarket_urgent"

        # Monday morning (good sign)
        if day == "Monday" and filing_time < time(11, 0):
            return "monday_morning"

        # Default: standard market hours
        return "market_hours"

    def analyze_filings_from_qdrant(
        self,
        ticker: str,
        days_back: int = 30
    ) -> List[FilingTimeSignal]:
        """
        Analyze filing times for signals already in Qdrant.

        Queries midge_signals collection for Form 4 and 8-K signals,
        then analyzes their filing timestamps.

        Args:
            ticker: Stock ticker to analyze
            days_back: Number of days to look back

        Returns:
            List of FilingTimeSignal objects
        """
        # Query Qdrant for recent filings
        try:
            response = requests.post(
                f"{self.qdrant_url}/collections/midge_signals/points/scroll",
                json={
                    "limit": 100,
                    "with_payload": True,
                    "filter": {
                        "must": [
                            {"key": "ticker", "match": {"value": ticker.upper()}},
                            {"key": "signal_source", "match": {"any": ["insider", "sec_form8k"]}}
                        ]
                    }
                },
                timeout=10
            )

            if response.status_code != 200:
                print(f"Qdrant query failed: {response.status_code}")
                return []

            data = response.json()
            points = data.get("result", {}).get("points", [])

            signals = []
            for point in points:
                payload = point.get("payload", {})

                # Skip if no filing date
                filing_date = payload.get("filing_date", "")
                if not filing_date:
                    continue

                # Try to parse datetime (assume noon if no time provided)
                try:
                    if "T" in filing_date:
                        filing_dt = datetime.fromisoformat(filing_date.replace("Z", ""))
                    else:
                        # Assume 4 PM ET (common filing time) if no time
                        filing_dt = datetime.strptime(f"{filing_date} 16:00", "%Y-%m-%d %H:%M")
                except Exception:
                    continue

                signal = self.analyze_filing_time(
                    ticker=payload.get("ticker", ticker),
                    filer_name=payload.get("filer_name", "Unknown"),
                    filing_date=filing_date,
                    filing_datetime=filing_dt,
                    form_type=payload.get("form_type", "4")
                )
                signals.append(signal)

            return signals

        except Exception as e:
            print(f"Error querying Qdrant: {e}")
            return []

    def get_historical_patterns(self, ticker: str) -> Dict[str, float]:
        """
        Analyze historical filing patterns for a ticker.

        Returns distribution of pattern types seen historically.

        Args:
            ticker: Stock ticker

        Returns:
            Dict mapping pattern_type to frequency (0-1)
        """
        signals = self.analyze_filings_from_qdrant(ticker, days_back=365)

        if not signals:
            return {}

        pattern_counts = {}
        for signal in signals:
            pt = signal.pattern_type
            pattern_counts[pt] = pattern_counts.get(pt, 0) + 1

        total = len(signals)
        return {pt: count / total for pt, count in pattern_counts.items()}


def analyze_filing_times(ticker: str, days_back: int = 30) -> List[FilingTimeSignal]:
    """
    Convenience function to analyze filing times for a ticker.

    Args:
        ticker: Stock ticker
        days_back: Number of days to look back

    Returns:
        List of FilingTimeSignal objects
    """
    analyzer = FilingTimeAnalyzer()
    return analyzer.analyze_filings_from_qdrant(ticker, days_back)


if __name__ == "__main__":
    import sys

    # Test with a sample filing
    print("Testing Filing Time Analyzer...")
    print()

    analyzer = FilingTimeAnalyzer()

    # Test various scenarios
    test_cases = [
        ("AAPL", "Tim Cook", "2026-02-06", datetime(2026, 2, 6, 16, 30)),  # Friday 4:30 PM (hiding)
        ("MSFT", "Satya Nadella", "2026-02-02", datetime(2026, 2, 2, 9, 15)),  # Monday 9:15 AM (routine)
        ("NVDA", "Jensen Huang", "2026-02-04", datetime(2026, 2, 4, 18, 0)),  # Wednesday after hours
        ("TSLA", "Elon Musk", "2026-02-05", datetime(2026, 2, 5, 8, 30)),  # Thursday pre-market
    ]

    for ticker, filer, date, dt in test_cases:
        signal = analyzer.analyze_filing_time(ticker, filer, date, dt)
        print(f"  {signal.to_plain_language()}")

    print()

    # Test with Qdrant if ticker provided
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        print(f"Analyzing {ticker} filings from Qdrant...")
        signals = analyze_filing_times(ticker)
        if signals:
            for signal in signals[:5]:
                print(f"  {signal.to_plain_language()}")
        else:
            print("  No filing data found in Qdrant")

    print("\nDone.")
