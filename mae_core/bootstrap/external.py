"""Bootstrap Layer 31: External API Gateway.

Creates: ApiGateway, registers holon, wires connections, and
registers available providers:
  - LLM providers (Groq, Mistral, DeepSeek, Claude)
  - Financial data providers (MarketAux, Finnhub, Alpha Vantage)
  - Web search provider (Tavily)

Biological analogy: Growing sensory organs. This layer gives Mae her
mouth and ears — the ability to ask questions of the external world
and receive answers. LLM providers are her voice (she can ask and be
answered). Data providers are her chemoreceptors (financial market
signals). Tavily is her olfactory sense (broad environmental scan).
The existing immune system (BoundaryMembrane, InputValidator) validates
everything that crosses the boundary.

Graceful degradation: If no API keys are set, the gateway exists but
has zero providers. Agents never see api_call tasks. The organism
runs exactly as before.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

logger = logging.getLogger("mae.bootstrap")


def bootstrap_external(ctx: SimpleNamespace) -> None:
    """Create external API gateway and register available providers (Layer 31)."""
    from mae_core.external.api_gateway import ApiGateway
    from mae_core.external.llm_provider import LlmProvider
    from mae_core.external.openai_provider import OpenAIProvider

    # =================================================================
    # Layer 31a: Create ApiGateway
    # =================================================================
    ctx.api_gateway = ApiGateway(
        event_bus=ctx.bus,
        boundary_membrane=ctx.boundary_membrane,
        input_validator=ctx.input_validator,
        task_pool=ctx.task_pool,
    )

    # Register as step hook (cadence handled by gateway internally)
    ctx.model.add_step_hook(
        lambda: ctx.api_gateway.step(current_step=int(ctx.model.time))
    )

    # Register with SomaticMap
    if hasattr(ctx.somatic_map, "register_system"):
        try:
            ctx.somatic_map.register_system(
                system_id="api_gateway",
                description="ApiGateway",
                depends_on=["boundary_membrane", "input_validator"],
            )
        except Exception:
            logger.debug("Could not register api_gateway with SomaticMap")

    # Register as holon
    ctx.holon_registry.register("api_gateway", holon_type="system", parent_id="mae")
    proxy = ctx.holon_registry.get_proxy("api_gateway")
    if proxy is not None:
        proxy.set_system_ref(ctx.api_gateway)

    logger.info(
        "Layer 31a - ApiGateway created (boundary_membrane=%s, input_validator=%s)",
        ctx.boundary_membrane is not None,
        ctx.input_validator is not None,
    )

    # =================================================================
    # Layer 31b: Register LLM providers (triadic: Groq + Mistral + DeepSeek)
    # =================================================================
    _register_llm_providers(ctx)

    # =================================================================
    # Layer 31c: Register data providers (MarketAux, Finnhub, Alpha Vantage, Tavily)
    # =================================================================
    _register_data_providers(ctx)

    # =================================================================
    # Layer 31d: Register triadic connections
    # =================================================================
    _register_external_connections(ctx)

    logger.info(
        "Layer 31 - External API Gateway complete (%d providers)",
        ctx.api_gateway.provider_count,
    )


def _register_llm_providers(ctx: SimpleNamespace) -> None:
    """Register available LLM providers with the ApiGateway.

    Checks environment for API keys and registers each available provider.
    Triadic design: Groq (speed), Mistral (quality), DeepSeek (reasoning).
    Claude retained as fallback if key is present.

    Groq supports 5-account rotation: MAE_GROQ_API_KEY (key 1) plus
    MAE_GROQ_API_KEY_2 through MAE_GROQ_API_KEY_5. Each gets its own
    provider instance (groq_1..groq_5). The ApiGateway round-robins
    requests with provider="groq" across all available groq_N providers.
    """
    from mae_core.external.openai_provider import OpenAIProvider
    from mae_core.external.llm_provider import LlmProvider

    # -- Groq: 5-account rotation --
    groq_base_url = "https://api.groq.com/openai/v1"
    groq_model = "llama-3.3-70b-versatile"
    groq_env_vars = [
        ("groq_1", "MAE_GROQ_API_KEY"),
        ("groq_2", "MAE_GROQ_API_KEY_2"),
        ("groq_3", "MAE_GROQ_API_KEY_3"),
        ("groq_4", "MAE_GROQ_API_KEY_4"),
        ("groq_5", "MAE_GROQ_API_KEY_5"),
    ]
    groq_names = []
    for name, env_var in groq_env_vars:
        api_key = os.environ.get(env_var, "")
        if api_key:
            provider = OpenAIProvider(
                name=name,
                api_key=api_key,
                base_url=groq_base_url,
                model=groq_model,
            )
            ctx.api_gateway.register_provider(name, provider)
            _trust_provider(ctx, name)
            groq_names.append(name)

    if groq_names:
        ctx.api_gateway.register_rotation("groq", groq_names)
        # Trust the rotation group name so BoundaryMembrane doesn't
        # quarantine requests using provider="groq" (the alias).
        _trust_provider(ctx, "groq")
        logger.info(
            "Layer 31b - Groq rotation: %d accounts (%s)",
            len(groq_names), ", ".join(groq_names),
        )

    # -- Other OpenAI-compatible providers --
    other_providers = [
        ("mistral", "MAE_MISTRAL_API_KEY", "https://api.mistral.ai/v1", "mistral-small-latest"),
        ("deepseek", "MAE_DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
    ]

    for name, env_var, base_url, model in other_providers:
        api_key = os.environ.get(env_var, "")
        if api_key:
            provider = OpenAIProvider(
                name=name,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
            ctx.api_gateway.register_provider(name, provider)
            _trust_provider(ctx, name)
            logger.info("Layer 31b - %s provider registered (model=%s)", name, model)

    # Claude provider (Anthropic-specific format)
    claude_key = os.environ.get("MAE_CLAUDE_API_KEY", "")
    if claude_key:
        claude_provider = LlmProvider(api_key=claude_key)
        ctx.api_gateway.register_provider("claude", claude_provider)
        _trust_provider(ctx, "claude")
        logger.info("Layer 31b - Claude provider registered (model=%s)", claude_provider._model)

    if ctx.api_gateway.provider_count == 0:
        logger.info(
            "Layer 31b - No LLM API keys found. "
            "ApiGateway has 0 providers (graceful degradation)"
        )


def _trust_provider(ctx: SimpleNamespace, name: str) -> None:
    """Register a provider as trusted with BoundaryMembrane."""
    if ctx.boundary_membrane is not None:
        try:
            if hasattr(ctx.boundary_membrane, "register_source"):
                ctx.boundary_membrane.register_source(name, trust=0.8)
        except Exception:
            logger.debug("Could not register %s with BoundaryMembrane", name)


def _register_data_providers(ctx: SimpleNamespace) -> None:
    """Register financial data and web search providers.

    Checks environment for API keys and registers each available provider.
    All use the BoundaryMembrane trust pathway via _trust_provider().

    Financial providers (chemoreceptors):
        - MarketAux: financial news with sentiment scoring
        - Finnhub: real-time market data (quotes, candles, company info)
        - Alpha Vantage: stock/crypto/forex time series

    Web search provider (olfactory):
        - Tavily: broad web search with AI-generated answers
    """
    from mae_core.external.rest_data_provider import RestDataProvider
    from mae_core.external.tavily_provider import TavilyProvider

    # Financial data providers — all authenticate via a query parameter
    financial_providers = [
        (
            "marketaux",
            "MAE_MARKETAUX_API_KEY",
            "https://api.marketaux.com/v1",
            "api_token",
            "/news/all",
        ),
        (
            "finnhub",
            "MAE_FINNHUB_API_KEY",
            "https://finnhub.io/api/v1",
            "token",
            "/quote",
        ),
        (
            "alphavantage",
            "MAE_ALPHAVANTAGE_API_KEY",
            "https://www.alphavantage.co",
            "apikey",
            "query",
        ),
    ]

    for name, env_var, base_url, auth_param, default_endpoint in financial_providers:
        api_key = os.environ.get(env_var, "")
        if api_key:
            provider = RestDataProvider(
                name=name,
                api_key=api_key,
                base_url=base_url,
                auth_param=auth_param,
                default_endpoint=default_endpoint,
            )
            ctx.api_gateway.register_provider(name, provider)
            _trust_provider(ctx, name)
            logger.info("Layer 31c - %s provider registered (%s)", name, base_url)

    # Web search provider
    tavily_key = os.environ.get("MAE_TAVILY_API_KEY", "")
    if tavily_key:
        tavily = TavilyProvider(api_key=tavily_key)
        ctx.api_gateway.register_provider("tavily", tavily)
        _trust_provider(ctx, "tavily")
        logger.info("Layer 31c - tavily provider registered (web search)")

    data_count = sum(
        1 for name in ("marketaux", "finnhub", "alphavantage", "tavily")
        if ctx.api_gateway.has_provider(name)
    )
    if data_count == 0:
        logger.info(
            "Layer 31c - No data API keys found. "
            "Financial/search providers inactive (graceful degradation)"
        )


def _register_external_connections(ctx: SimpleNamespace) -> None:
    """Register Group 12: External API connections (Law 1 compliant)."""
    from mae_core.backbone.connection_registry import (
        ConnectionType,
        ConnectionCriticality,
    )

    reg = ctx.connection_registry.register_connection
    dr = ConnectionType.DIRECT_REFERENCE
    eb = ConnectionType.EVENTBUS_PUBSUB
    sh = ConnectionType.STEP_HOOK

    # ApiGateway -> BoundaryMembrane (direct: gateway calls check_input)
    reg("api_gateway", "boundary_membrane", dr,
        witnesses=["input_validator", "threat_detector"],
        criticality=ConnectionCriticality.IMPORTANT,
        description="Gateway validates sources through membrane")

    # ApiGateway -> InputValidator (direct: gateway calls validate)
    reg("api_gateway", "input_validator", dr,
        witnesses=["boundary_membrane", "threat_detector"],
        criticality=ConnectionCriticality.IMPORTANT,
        description="Gateway validates payloads and responses")

    # ApiGateway -> EventBus: response received
    reg("api_gateway", "event_bus", eb,
        channel="external.response_received",
        witnesses=["boundary_membrane", "input_validator"],
        description="External response enters Mae's consciousness")

    # ApiGateway -> EventBus: request queued
    reg("api_gateway", "event_bus", eb,
        channel="external.request_queued",
        witnesses=["boundary_membrane", "auditor"],
        description="External request queued for processing")

    # ApiGateway -> EventBus: request failed
    reg("api_gateway", "event_bus", eb,
        channel="external.request_failed",
        witnesses=["threat_detector", "auto_healer"],
        criticality=ConnectionCriticality.IMPORTANT,
        description="External request failure — defense + healing aware")

    # Agent -> ApiGateway: submit request (via EventBus)
    reg("agent", "api_gateway", eb,
        channel="external.submit_request",
        witnesses=["boundary_membrane", "input_validator"],
        description="Agent submits external consultation request")

    # ApiGateway -> TaskPool (direct: completes tasks with responses)
    reg("api_gateway", "task_pool", dr,
        witnesses=["boundary_membrane", "input_validator"],
        description="Gateway completes tasks with API responses")

    # ApiGateway -> Model (step hook: periodic queue processing)
    reg("api_gateway", "model", sh,
        witnesses=["boundary_membrane", "auditor"],
        description="Gateway processes queue each step")

    logger.info("Group 12 - External Connections: 8 triadic connections registered")
