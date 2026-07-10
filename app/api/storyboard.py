"""Storyboard router.

PORT NOTE: The Next.js API has no dedicated storyboard route file. Storyboard
*versions* are created by the shot-split generation flow (see generate.py) and
are served inline as the ``versions`` array of the project- and episode-detail
GET endpoints (projects.py / episodes.py). This module exists to keep the
one-router-per-resource layout and as the home for any future standalone
storyboard endpoints; it currently exposes no routes.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
