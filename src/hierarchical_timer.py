"""Hierarchical timing wheel for scheduling callbacks at future ticks.

Two-level wheel: level-0 has 8 slots (ticks 0-7), level-1 has 8 slots
(each covering 8 ticks). Total range: 0..63 (MAX_DELAY - 1).
Capacity: MAX_DELAY entries total across both levels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

MAX_DELAY = 64
_LEVEL_SIZE = 8


@dataclass
class _TimerEntry:
    remaining: int
    callback: Callable[[], None]
    cancelled: bool = False
    id: int = 0


class HierarchicalTimer:
    """Two-level hierarchical timing wheel."""

    def __init__(self) -> None:
        self._level0: list[list[_TimerEntry]] = [[] for _ in range(_LEVEL_SIZE)]
        self._level1: list[list[_TimerEntry]] = [[] for _ in range(_LEVEL_SIZE)]
        self._tick: int = 0
        self._count: int = 0
        self._next_id: int = 0

    def __len__(self) -> int:
        return self._count

    def schedule(self, delay: int, callback: Callable[[], None]) -> int:
        """Schedule *callback* to fire after *delay* ticks. Returns timer id."""
        if delay < 0 or delay >= MAX_DELAY:
            raise ValueError(f"delay {delay} out of range [0, {MAX_DELAY})")
        if self._count >= MAX_DELAY:
            raise RuntimeError("timer capacity exceeded")

        entry = _TimerEntry(remaining=delay, callback=callback)
        self._next_id += 1
        entry.id = self._next_id
        self._insert(entry)
        self._count += 1
        return entry.id

    def _insert(self, entry: _TimerEntry) -> None:
        delay = entry.remaining
        if delay < _LEVEL_SIZE:
            # delay=0 means fire on the very next tick, so +1
            slot = (self._tick + max(delay, 1)) % _LEVEL_SIZE
            self._level0[slot].append(entry)
        else:
            l1_slot = delay // _LEVEL_SIZE
            if l1_slot >= _LEVEL_SIZE:
                l1_slot = _LEVEL_SIZE - 1
            self._level1[l1_slot].append(entry)

    def tick(self) -> None:
        """Advance one tick and fire due callbacks."""
        self._tick += 1
        slot = self._tick % _LEVEL_SIZE
        # If we wrapped around level-0, cascade from level-1
        if slot == 0:
            self._cascade()
        # Fire everything in the current level-0 slot
        entries = self._level0[slot]
        self._level0[slot] = []
        for entry in entries:
            if not entry.cancelled:
                entry.callback()
                self._count -= 1

    def _cascade(self) -> None:
        """Move entries from level-1 into level-0."""
        l1_slot = (self._tick // _LEVEL_SIZE) % _LEVEL_SIZE
        entries = self._level1[l1_slot]
        self._level1[l1_slot] = []
        for entry in entries:
            if entry.cancelled:
                continue
            # Recalculate remaining ticks relative to current tick
            entry.remaining = entry.remaining % _LEVEL_SIZE
            slot = (self._tick + entry.remaining) % _LEVEL_SIZE
            self._level0[slot].append(entry)

    def run_until(self, target_tick: int) -> None:
        """Tick until we've reached *target_tick* total ticks."""
        while self._tick < target_tick:
            self.tick()

    def cancel(self, timer_id: int) -> bool:
        """Cancel a scheduled timer. Returns True if it was still pending."""
        for level in (self._level0, self._level1):
            for slot in level:
                for entry in slot:
                    if entry.id == timer_id and not entry.cancelled:
                        entry.cancelled = True
                        self._count -= 1
                        return True
        return False
