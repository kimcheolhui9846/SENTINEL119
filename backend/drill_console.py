"""Serve the existing console and the independent virtual training sessions."""

from .drill_router import make_drill_router
from .team_demo import create_console as create_manual_console


def create_console():
    app = create_manual_console()
    app.include_router(make_drill_router())
    return app
