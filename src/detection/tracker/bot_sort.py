from __future__ import annotations

from collections import OrderedDict
from typing import Any
from uuid import uuid4

import numpy as np

from src.detection.tracker import matching
from src.detection.tracker.kalman_filter import KalmanFilterXYWH


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0
    _unique_id: str = None

    def __init__(self):
        self.track_id = 0
        self.track_unique_id = ""
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0.0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        BaseTrack._unique_id = str(uuid4())
        return BaseTrack._count, BaseTrack._unique_id

    def activate(self, *args: Any) -> None:
        raise NotImplementedError

    def predict(self) -> None:
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    @staticmethod
    def reset_id() -> None:
        BaseTrack._count = 0


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlbr: np.ndarray | list[float], score: float, cls: Any):
        super().__init__()
        self._tlbr = np.asarray(tlbr[:4], dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None

        self.score = float(score)
        self.tracklet_len = 0
        self.cls = cls

    def predict(self) -> None:
        if self.mean is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[4:] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[STrack]) -> None:
        if not stracks:
            return

        valid_tracks = [st for st in stracks if st.mean is not None and st.covariance is not None]
        if not valid_tracks:
            return

        multi_mean = np.asarray([st.mean.copy() for st in valid_tracks], dtype=np.float32)
        multi_covariance = np.asarray([st.covariance for st in valid_tracks], dtype=np.float32)

        for i, st in enumerate(valid_tracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][4:] = 0

        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

        for st, mean, cov in zip(valid_tracks, multi_mean, multi_covariance):
            st.mean = mean
            st.covariance = cov

    def activate(self, kalman_filter: KalmanFilterXYWH, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id, self.track_unique_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlbr_to_xywh(self._tlbr))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self.tlbr_to_xywh(new_track.xyxy),
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

        if new_id:
            self.track_id, self.track_unique_id = self.next_id()

        self.score = new_track.score
        self.cls = new_track.cls

    def update(self, new_track: STrack, frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self.tlbr_to_xywh(new_track.xyxy),
        )

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self.tlbr_to_tlwh(self._tlbr)
        ret = self.mean[:4].copy()  # xywh
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xywh(self) -> np.ndarray:
        if self.mean is None:
            return self.tlbr_to_xywh(self._tlbr)
        return self.mean[:4].copy()

    @property
    def xyxy(self) -> np.ndarray:
        if self.mean is None:
            return self._tlbr.copy()
        x, y, w, h = self.mean[:4]
        return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)

    @property
    def result(self) -> list[float]:
        coords = self.xyxy
        return [*coords.tolist(), self.cls, self.score, self.track_id, self.track_unique_id]

    @staticmethod
    def tlbr_to_xywh(tlbr: np.ndarray) -> np.ndarray:
        tlbr = np.asarray(tlbr, dtype=np.float32)
        x1, y1, x2, y2 = tlbr[:4]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return np.array([cx, cy, w, h], dtype=np.float32)

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        tlbr = np.asarray(tlbr, dtype=np.float32)
        x1, y1, x2, y2 = tlbr[:4]
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        tlwh = np.asarray(tlwh, dtype=np.float32)
        ret = tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret

    def __repr__(self) -> str:
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BOTrack(STrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlbr: np.ndarray, score: float, cls: int):
        super().__init__(tlbr, score, cls)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9

    @staticmethod
    def multi_predict(stracks: list[BOTrack]) -> None:
        STrack.multi_predict(stracks)


class BOTSORT:
    def __init__(self, args: Any, frame_rate: int = 20):
        self.tracked_stracks: list[BOTrack] = []
        self.lost_stracks: list[BOTrack] = []
        self.removed_stracks: list[BOTrack] = []

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 20 * args.track_buffer)
        self.kalman_filter = KalmanFilterXYWH()
        self.reset_id()

        self.tracker_id_count = 0

    def update(self, bboxes, cls, scores) -> np.ndarray:
        self.frame_id += 1

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        bboxes = np.asarray(bboxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        cls = np.asarray(cls, dtype=np.int32)

        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        inds_second = inds_low & inds_high

        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        cls_keep = cls[remain_inds]

        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]
        cls_second = cls[inds_second]

        detections = self.init_track(dets, scores_keep, cls_keep)

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        detections_second = self.init_track(dets_second, scores_second, cls_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track_second, _ = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        for it in u_track_second:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            self.tracker_id_count += 1
            track.track_id = self.tracker_id_count
            track.track_unique_id = str(uuid4())
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, removed_stracks)

        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        self.removed_stracks.extend(removed_stracks)

        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        # return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)
        return [x.result for x in self.tracked_stracks if x.is_activated]

    def init_track(self, dets, scores, cls) -> list[BOTrack]:
        if len(dets) == 0:
            return []
        return [BOTrack(tlbr, s, c) for tlbr, s, c in zip(dets, scores, cls)]

    def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks: list[BOTrack]) -> None:
        BOTrack.multi_predict(tracks)

    def reset_id(self) -> None:
        STrack.reset_id()

    def reset(self) -> None:
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYWH()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
