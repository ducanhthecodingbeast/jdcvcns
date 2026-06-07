from __future__ import annotations

import os


def is_configured() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))


def get_client():
    if not is_configured():
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required for Supabase persistence.")
    from supabase import create_client

    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
