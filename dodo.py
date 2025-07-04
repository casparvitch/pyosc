import os

from doit.action import CmdAction


def task_format():
    """Format code using ruff."""

    def router(help=False):
        if help:
            return """echo '
Code Formatter Help
=================

This task runs the ruff formatter to ensure consistent code style:
- Sorts imports (ruff check --select I --fix)
- Formats code (ruff format)

No options required - simply run:
  doit format
  '"""
        return "ruff check --select I --fix . && ruff format . "

    return {
        "actions": [CmdAction(router)],
        "params": [
            {
                "name": "help",
                "long": "help",
                "default": False,
                "type": bool,
            },
        ],
        "verbosity": 2,
    }
