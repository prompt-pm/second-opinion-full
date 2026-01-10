from phoenix.otel import register
import os


def set_hosted_phoenix_instrumentation():
    """Set tracing instrumentation for Phoenix and Arize"""
    PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY")
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
    os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
    # Setup OTEL tracing for hosted Phoenix. The register function will automatically detect the endpoint and headers from your environment variables.
    tracer_provider = register(batch=True, auto_instrument=True)
    print("Tracing instrumentation set")
