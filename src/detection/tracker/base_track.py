from enum import IntEnum


class TrackState(IntEnum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count

    @classmethod
    def reset_id(cls):
        cls._count = 0
