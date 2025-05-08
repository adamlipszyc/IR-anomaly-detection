import logging
from typing import Type, Callable
from functools import wraps
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table

# Configure root logger to use Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

def make_summary(title: str, stats: dict) -> None:
    """
    Display a summary table of task outcomes.
    """
    console = Console()
    table = Table(title=title)
    table.add_column("Task", style="cyan")
    table.add_column("Status", style="magenta", justify="right")
    for task, status in stats.items():
        table.add_row(task, status)
    console.print(table)



def catch_and_log(exception: Type[Exception], action: str = "") -> Callable:
    """
    Decorator to catch specified exception, log error, and re-raise.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_logger = getattr(args[0], "logger", logger) if args else logger
            try:
                return func(*args, **kwargs)
            except exception as e:
                current_logger.error("Error %s[%s]: %s", action or func.__name__, type(e).__name__, e)
                raise
            finally:
                current_logger.info(f"{func.__name__}")
        return wrapper
    return decorator