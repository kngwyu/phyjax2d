from __future__ import annotations

import pathlib
import shutil
import subprocess

import nox

SOURCES = ["src/phyjax2d", "tests"]


@nox.session(reuse_venv=True)
def format(session: nox.Session) -> None:
    session.install("black")
    session.install("isort")
    session.run("black", *SOURCES)
    session.run("isort", *SOURCES)


@nox.session(reuse_venv=True, python=["3.10", "3.11", "3.12"])
def lint(session: nox.Session) -> None:
    session.install("ruff")
    session.install("black")
    session.install("isort")
    session.run("ruff", "check", *SOURCES)
    session.run("black", *SOURCES, "--check")
    session.run("isort", *SOURCES, "--check")


@nox.session(reuse_venv=True)
def publish(session: nox.Session) -> None:
    session.install("build")
    session.install("twine")
    session.run("python", "-m", "build")
    session.run("python", "-m", "twine", "upload", "--repository", "pypi", "dist/*")


@nox.session(reuse_venv=True, python=["3.10", "3.11", "3.12"])
def tests(session: nox.Session) -> None:
    session.install("-e", ".")
    session.install("pytest")
    session.run("pytest", "tests", *session.posargs)
