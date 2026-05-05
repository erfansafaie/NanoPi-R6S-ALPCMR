import numpy as np
from base_track import BaseTrack, TrackState


class STrack(BaseTrack):
    __slots__ = (
        "mean",
        "covariance",
        "kalman_filter",
        "score",
        "cls",
        "track_id",
        "state",
        "is_activated",
        "frame_id",
        "start_frame",
        "time_since_update",
        "_tlbr_init",
        "_xywh_init",
    )

    def __init__(self, tlbr, score, cls_id):
        self.mean = None
        self.covariance = None
        self.kalman_filter = None

        self.score = float(score)
        self.cls = int(cls_id)

        self.track_id = 0
        self.state = TrackState.New
        self.is_activated = False

        self.frame_id = 0
        self.start_frame = 0
        self.time_since_update = 0

        self._tlbr_init = np.asarray(tlbr, dtype=np.float32).copy()
        self._xywh_init = self.tlbr_to_xywh(self._tlbr_init)

    @staticmethod
    def tlbr_to_xywh(tlbr):
        x1, y1, x2, y2 = tlbr
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return np.array([cx, cy, w, h], dtype=np.float32)

    @staticmethod
    def xywh_to_tlbr(xywh):
        cx, cy, w, h = xywh
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @property
    def tlbr(self):
        if self.mean is not None:
            return self.xywh_to_tlbr(self.mean[:4])
        return self._tlbr_init.copy()

    @property
    def xywh(self):
        if self.mean is not None:
            return self.mean[:4].copy()
        return self._xywh_init.copy()

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self._xywh_init)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0

    def predict(self):
        if self.mean is not None:
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.time_since_update = 0
        self.score = float(new_track.score)
        self.cls = int(new_track.cls)

        measurement = new_track.xywh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement
        )
        self.state = TrackState.Tracked
        self.is_activated = True

    def re_activate(self, new_track, frame_id, new_id=False):
        self.frame_id = frame_id
        self.time_since_update = 0
        self.score = float(new_track.score)
        self.cls = int(new_track.cls)

        measurement = new_track.xywh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        if new_id:
            self.track_id = self.next_id()

    def mark_lost(self):
        self.state = TrackState.Lost
        self.is_activated = False

    def mark_removed(self):
        self.state = TrackState.Removed
        self.is_activated = False
