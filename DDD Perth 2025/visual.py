"""
visual.py
Shared plotting helpers for queue simulations.
Keep visuals high-contrast and readable on projectors.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def plot_queue_and_waits(queue_len: List[Tuple[float, int]],
                         waits_over_time: List[Tuple[float, float]],
                         title: str = "Coffee shop simulation",
                         figsize: Tuple[int, int] = (9, 6),
                         show: bool = True,
                         save_path: Optional[str] = None) -> None:
    """
    Plot queue length over time (top) and individual wait times (bottom).
    queue_len: list of (time, queue_length)
    waits_over_time: list of (time_of_service_end, wait_duration)
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Queue length plot
    if queue_len:
        t_q, q = zip(*queue_len)
        axes[0].step(t_q, q, where="post", color="#1f77b4")
    axes[0].set_ylabel("Queue length")
    axes[0].grid(True, alpha=0.3)

    # Waits plot
    if waits_over_time:
        t_w, w = zip(*waits_over_time)
        axes[1].plot(t_w, w, ".", alpha=0.75, color="#d62728")
    axes[1].set_ylabel("Wait (min)")
    axes[1].set_xlabel("Time (min)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)