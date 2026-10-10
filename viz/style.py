"""The ONE palette and style for every figure of the project. Two color vocabularies, deliberately
separate and never mixed within the same panel: the semantic one (prior / posterior / truth / miss) and
the one by channel.
"""

import matplotlib.pyplot as plt

# --- Ink / surface ---------------------------------------------------------
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# --- Semantic: prior / posterior / truth / miss ----------------------------
PRIOR_COLOR = "#2a78d6"       # blue
POSTERIOR_COLOR = "#eb6834"   # orange
TRUTH_COLOR = INK             # black, always a dashed line
MISS_COLOR = "#c0392b"        # red -- RESERVED for "does not cover HDI94", never
                               # reused as a channel color or anything else.

# --- By channel (fixed identity across the WHOLE project) -----------------
CHANNEL_COLORS: dict[str, str] = {
    "tv_impressions_000": "#2a78d6",
    "meta_impressions_000": "#eb6834",
    "tiktok_impressions_000": "#1baf7a",
    "ooh_impressions_000": "#8a5fbf",
}


def channel_color(channel: str) -> str:
    """Accepts both 'tv_impressions_000' and 'tv' (short name)."""
    if channel in CHANNEL_COLORS:
        return CHANNEL_COLORS[channel]
    full = f"{channel}_impressions_000"
    if full in CHANNEL_COLORS:
        return CHANNEL_COLORS[full]
    raise KeyError(f"Unknown channel: {channel!r}. Known: {list(CHANNEL_COLORS)}")


def channel_short_name(channel_col: str) -> str:
    return channel_col.replace("_impressions_000", "")


def apply_style() -> None:
    """Applies the project's single rcParams. Call it once at the start of each figure script, before
    creating any Figure.
    """
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY, "text.color": INK, "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED, "grid.color": GRID, "font.family": "sans-serif", "font.size": 9,  # cosmetic
    })
