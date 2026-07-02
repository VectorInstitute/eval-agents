"""Shared instance of langfuse client."""

from os import getenv

from langfuse import Langfuse
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..env_vars import Configs


__all__ = ["langfuse_client"]


config = Configs.from_env_var()
# Only initialize Langfuse if keys are provided. Some environments won't use tracing.
lf_pub = config.langfuse_public_key
lf_sec = config.langfuse_secret_key
if lf_pub and lf_sec:
    langfuse_client = Langfuse(public_key=lf_pub, secret_key=lf_sec)
else:
    langfuse_client = None


def flush_langfuse(client: "Langfuse | None" = None):
    """Flush shared LangFuse Client. Rich Progress included."""
    if client is None:
        client = langfuse_client

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Finalizing Langfuse annotations...", total=None)
        if client is not None:
            client.flush()
