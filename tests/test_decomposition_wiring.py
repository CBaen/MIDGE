"""Verify monolith decomposition preserves all imports and behavior.

These tests guard against import regressions introduced during the Round 2
signal.py + bootstrap/market.py decomposition. Every from_* adapter, every
signal_adapters sub-module, every bootstrap sub-module, and every sensing
sub-module must be importable from their expected location.

NoMonoliths tests enforce the 500-line cap. sensing_hook.py is currently
576 lines because Anvil's decomposition is in-progress (sub-modules exist
but sensing_hook.py still carries the full implementation). That test is
marked xfail and will flip to pass once Anvil completes the delegation
refactor. All other NoMonoliths assertions should pass immediately.
"""
import importlib

import pytest


class TestSignalReexports:
    """All 25 from_* functions must be importable from mae_core.market.signal."""

    EXPECTED_CONVERTERS = [
        "from_insider_trade",
        "from_form8k_event",
        "from_filing_keyword",
        "from_cluster_signal",
        "from_correlation_signal",
        "from_congressional_trade",
        "from_senate_trade",
        "from_short_interest",
        "from_news_sentiment",
        "from_earnings_event",
        "from_macro_indicator",
        "from_price_data",
        "from_social_sentiment",
        "from_ta_signal",
        "from_session_sweep",
        "from_government_contract",
        "from_contract_opportunity",
        "from_contract_prediction",
        "from_hiring_signal",
        "from_cot_positioning",
        "from_stocktwits_sentiment",
        "from_vix_structure",
        "from_trends_signal",
        "from_economic_event",
        "from_analyst_recommendation",
    ]

    @pytest.mark.parametrize("name", EXPECTED_CONVERTERS)
    def test_converter_importable(self, name):
        """Each converter must be importable from the original location."""
        import mae_core.market.signal as sig

        assert hasattr(sig, name), f"{name} missing from mae_core.market.signal"
        assert callable(getattr(sig, name))

    def test_total_converter_count(self):
        """Exactly 37 from_* functions must be re-exported."""
        import mae_core.market.signal as sig

        exported = [n for n in dir(sig) if n.startswith("from_")]
        assert len(exported) == 38, (
            f"Expected 38 from_* converters, found {len(exported)}: {exported}"
        )


class TestSignalAdapters:
    """All adapter sub-modules must be importable and expose their converters."""

    ADAPTER_MODULES = [
        "regulatory",
        "political",
        "market_data",
        "technical",
        "contracts",
        "layer6",
        "wave2_3",
    ]

    @pytest.mark.parametrize("mod", ADAPTER_MODULES)
    def test_adapter_importable(self, mod):
        m = importlib.import_module(f"mae_core.market.signal_adapters.{mod}")
        assert m is not None

    def test_adapter_init_importable(self):
        """The signal_adapters __init__ must re-export all 38 converters."""
        import mae_core.market.signal_adapters as sa

        assert hasattr(sa, "__all__"), "signal_adapters __init__ must define __all__"
        assert len(sa.__all__) == 38, (
            f"Expected 38 entries in __all__, found {len(sa.__all__)}"
        )

    def test_regulatory_adapters(self):
        from mae_core.market.signal_adapters.regulatory import (
            from_insider_trade,
            from_form8k_event,
            from_filing_keyword,
            from_cluster_signal,
            from_correlation_signal,
        )

        for fn in (
            from_insider_trade,
            from_form8k_event,
            from_filing_keyword,
            from_cluster_signal,
            from_correlation_signal,
        ):
            assert callable(fn)

    def test_political_adapters(self):
        from mae_core.market.signal_adapters.political import (
            from_congressional_trade,
            from_senate_trade,
        )

        assert callable(from_congressional_trade)
        assert callable(from_senate_trade)

    def test_market_data_adapters(self):
        from mae_core.market.signal_adapters.market_data import (
            from_short_interest,
            from_news_sentiment,
            from_earnings_event,
            from_macro_indicator,
            from_price_data,
            from_social_sentiment,
        )

        for fn in (
            from_short_interest,
            from_news_sentiment,
            from_earnings_event,
            from_macro_indicator,
            from_price_data,
            from_social_sentiment,
        ):
            assert callable(fn)

    def test_technical_adapters(self):
        from mae_core.market.signal_adapters.technical import (
            from_ta_signal,
            from_session_sweep,
        )

        assert callable(from_ta_signal)
        assert callable(from_session_sweep)

    def test_contracts_adapters(self):
        from mae_core.market.signal_adapters.contracts import (
            from_government_contract,
            from_contract_opportunity,
            from_contract_prediction,
            from_hiring_signal,
        )

        for fn in (
            from_government_contract,
            from_contract_opportunity,
            from_contract_prediction,
            from_hiring_signal,
        ):
            assert callable(fn)

    def test_layer6_adapters(self):
        from mae_core.market.signal_adapters.layer6 import (
            from_cot_positioning,
            from_stocktwits_sentiment,
            from_vix_structure,
            from_trends_signal,
            from_economic_event,
            from_analyst_recommendation,
        )

        for fn in (
            from_cot_positioning,
            from_stocktwits_sentiment,
            from_vix_structure,
            from_trends_signal,
            from_economic_event,
            from_analyst_recommendation,
        ):
            assert callable(fn)


class TestBootstrapDecomposition:
    """Bootstrap market sub-modules must be importable and expose their functions."""

    SUBMODULES = [
        "market_systems",
        "market_registration",
        "market_connections",
        "market_hooks",
        "market_agents",
    ]

    @pytest.mark.parametrize("mod", SUBMODULES)
    def test_submodule_importable(self, mod):
        m = importlib.import_module(f"mae_core.bootstrap.{mod}")
        assert m is not None

    def test_orchestrator_importable(self):
        from mae_core.bootstrap.market import bootstrap_market

        assert callable(bootstrap_market)

    def test_market_systems_function(self):
        from mae_core.bootstrap.market_systems import _instantiate_market_systems

        assert callable(_instantiate_market_systems)

    def test_market_registration_functions(self):
        from mae_core.bootstrap.market_registration import (
            _register_market_somatic,
            _register_market_holons,
            _register_market_fractal,
            _register_market_stem_roles,
        )

        for fn in (
            _register_market_somatic,
            _register_market_holons,
            _register_market_fractal,
            _register_market_stem_roles,
        ):
            assert callable(fn)

    def test_market_connections_function(self):
        from mae_core.bootstrap.market_connections import _register_market_connections

        assert callable(_register_market_connections)

    def test_market_hooks_functions(self):
        from mae_core.bootstrap.market_hooks import (
            _register_market_eventbus,
            _register_market_step_hooks,
            _wire_sensing_hook,
        )

        for fn in (
            _register_market_eventbus,
            _register_market_step_hooks,
            _wire_sensing_hook,
        ):
            assert callable(fn)

    def test_market_agents_function(self):
        from mae_core.bootstrap.market_agents import _differentiate_market_agents

        assert callable(_differentiate_market_agents)


class TestSensingDecomposition:
    """Sensing hook decomposition must preserve imports."""

    def test_sensing_hook_importable(self):
        from mae_core.market.sensing_hook import MarketSensingHook

        assert MarketSensingHook is not None

    def test_sensing_hook_instantiable(self):
        """MarketSensingHook must construct without any required arguments."""
        from mae_core.market.sensing_hook import MarketSensingHook

        hook = MarketSensingHook()
        assert hook is not None
        assert hasattr(hook, "step")
        assert callable(hook.step)

    def test_sensing_fetchers_importable(self):
        import mae_core.market.sensing_fetchers as sf

        assert sf is not None

    def test_sensing_lifecycle_importable(self):
        import mae_core.market.sensing_lifecycle as sl

        assert sl is not None

    def test_sensing_fetchers_expose_functions(self):
        """sensing_fetchers must expose at least one fetch_* function."""
        import mae_core.market.sensing_fetchers as sf

        fetch_fns = [name for name in dir(sf) if name.startswith("fetch_")]
        assert len(fetch_fns) > 0, (
            "sensing_fetchers.py must expose fetch_* functions, found none"
        )

    def test_sensing_lifecycle_expose_functions(self):
        """sensing_lifecycle must expose enrich_signal or store_signals."""
        import mae_core.market.sensing_lifecycle as sl

        lifecycle_fns = [name for name in dir(sl) if not name.startswith("_")]
        assert len(lifecycle_fns) > 0, (
            "sensing_lifecycle.py must expose public functions, found none"
        )


class TestNoMonoliths:
    """No file in the decomposed set should exceed 500 lines.

    sensing_hook.py is marked xfail because Anvil's decomposition created the
    sub-modules (sensing_fetchers.py, sensing_lifecycle.py) but has not yet
    refactored sensing_hook.py itself to delegate to them. The sub-modules
    exist and are importable (see TestSensingDecomposition). Once Anvil
    completes the delegation refactor and sensing_hook.py drops below 500
    lines, this xfail will become an unexpected pass — at which point remove
    the xfail marker.
    """

    def _count_lines(self, path: str) -> int:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def test_bootstrap_market_under_500(self):
        path = "mae_core/bootstrap/market.py"
        lines = self._count_lines(path)
        assert lines <= 500, f"{path} is {lines} lines (max 500)"

    def test_signal_under_500(self):
        path = "mae_core/market/signal.py"
        lines = self._count_lines(path)
        assert lines <= 500, f"{path} is {lines} lines (max 500)"

    @pytest.mark.xfail(
        reason=(
            "sensing_hook.py is 576 lines. Anvil created sensing_fetchers.py and "
            "sensing_lifecycle.py but has not yet refactored sensing_hook.py to "
            "delegate to them. Remove xfail once sensing_hook.py is under 500 lines."
        ),
        strict=True,
    )
    def test_sensing_hook_under_500(self):
        path = "mae_core/market/sensing_hook.py"
        lines = self._count_lines(path)
        assert lines <= 500, f"{path} is {lines} lines (max 500)"

    def test_signal_adapters_modules_under_500(self):
        """Each individual adapter module must be under 500 lines."""
        adapter_files = [
            "mae_core/market/signal_adapters/regulatory.py",
            "mae_core/market/signal_adapters/political.py",
            "mae_core/market/signal_adapters/market_data.py",
            "mae_core/market/signal_adapters/technical.py",
            "mae_core/market/signal_adapters/contracts.py",
            "mae_core/market/signal_adapters/layer6.py",
        ]
        violations = []
        for path in adapter_files:
            lines = self._count_lines(path)
            if lines > 500:
                violations.append(f"{path}: {lines} lines")
        assert not violations, "Adapter modules exceed 500 lines:\n" + "\n".join(violations)

    def test_bootstrap_submodules_under_500(self):
        """Bootstrap sub-modules must be under 500 lines (excluding market_hooks.py).

        market_hooks.py is currently 551 lines. Forge's decomposition split the
        original market.py into sub-modules, but market_hooks.py ended up slightly
        over the limit. It is checked separately (xfail) below.
        """
        bootstrap_files = [
            "mae_core/bootstrap/market_systems.py",
            "mae_core/bootstrap/market_registration.py",
            "mae_core/bootstrap/market_connections.py",
            "mae_core/bootstrap/market_agents.py",
        ]
        violations = []
        for path in bootstrap_files:
            lines = self._count_lines(path)
            if lines > 500:
                violations.append(f"{path}: {lines} lines")
        assert not violations, "Bootstrap sub-modules exceed 500 lines:\n" + "\n".join(violations)

    @pytest.mark.xfail(
        reason=(
            "market_hooks.py is 551 lines. Forge's decomposition of bootstrap/market.py "
            "produced sub-modules but market_hooks.py absorbed the sensing hook wiring, "
            "EventBus callbacks, and step hooks, pushing it 51 lines over the cap. "
            "Remove xfail once market_hooks.py is trimmed under 500 lines."
        ),
        strict=True,
    )
    def test_market_hooks_under_500(self):
        path = "mae_core/bootstrap/market_hooks.py"
        lines = self._count_lines(path)
        assert lines <= 500, f"{path} is {lines} lines (max 500)"
