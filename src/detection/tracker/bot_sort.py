import numpy as np

from base_track import TrackState
from kalman_filter import KalmanFilterXYWH
from matching import iou_distance, greedy_match
from track import STrack


class BoTSORT:
    """
    lightweight BoT-SORT-like tracker for embedded use.

    """

    def __init__(
        self,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        match_thresh=0.7,
        low_match_thresh=0.8,
        reactivate_match_thresh=0.7,
        track_buffer=30,
        frame_rate=30,
        class_aware=True,
    ):
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)

        self.match_thresh = float(match_thresh)
        self.low_match_thresh = float(low_match_thresh)
        self.reactivate_match_thresh = float(reactivate_match_thresh)

        self.buffer_size = int(track_buffer)
        self.frame_rate = int(frame_rate)
        self.class_aware = bool(class_aware)

        self.kalman_filter = KalmanFilterXYWH()

        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0

    def _parse_inputs(self, boxes, scores, classes):
        if boxes is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if scores is None:
            scores = np.zeros((0,), dtype=np.float32)
        if classes is None:
            classes = np.zeros((0,), dtype=np.int32)

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        classes = np.asarray(classes).reshape(-1)

        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)

        if classes.dtype.kind not in ("i", "u"):
            classes = classes.astype(np.int32)
        else:
            classes = classes.astype(np.int32, copy=False)

        assert boxes.ndim == 2 and boxes.shape[1] == 4, "boxes must be shape (N,4)"
        assert scores.ndim == 1, "scores must be shape (N,)"
        assert classes.ndim == 1, "classes must be shape (N,)"
        assert len(boxes) == len(scores) == len(classes), "boxes/scores/classes length mismatch"

        return boxes, scores, classes

    def _split_detections(self, boxes, scores, classes):
        high_dets = []
        low_dets = []

        for i in range(len(boxes)):
            score = float(scores[i])
            if score < self.track_low_thresh:
                continue

            det = STrack(boxes[i], score, int(classes[i]))

            if score >= self.track_high_thresh:
                high_dets.append(det)
            else:
                low_dets.append(det)

        return high_dets, low_dets

    def _predict_tracks(self, tracks):
        for t in tracks:
            t.predict()

    def _remove_duplicate_ids_keep_first(self, tracks):
        out = []
        seen = set()
        for t in tracks:
            if t.track_id in seen:
                continue
            seen.add(t.track_id)
            out.append(t)
        return out

    def update(self, boxes, scores, classes):
        """
        Args:
            boxes:   ndarray (N,4) in xyxy/tlbr
            scores:  ndarray (N,)
            classes: ndarray (N,)

        Returns:
            active tracked objects list[STrack]
        """
        self.frame_id += 1

        boxes, scores, classes = self._parse_inputs(boxes, scores, classes)
        high_dets, low_dets = self._split_detections(boxes, scores, classes)

        # Predict tracked and lost tracks forward
        self._predict_tracks(self.tracked_stracks)
        self._predict_tracks(self.lost_stracks)

        activated_tracks = []
        reactivated_tracks = []
        newly_lost_tracks = []
        newly_removed_tracks = []

        # --------------------------------------------------
        # Stage 1: current tracked tracks vs high detections
        # --------------------------------------------------
        tracked_pool = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]

        cost_1 = iou_distance(tracked_pool, high_dets, class_aware=self.class_aware)
        matches_1, u_tracked_1, u_high_1 = greedy_match(cost_1, self.match_thresh)

        for track_idx, det_idx in matches_1:
            track = tracked_pool[track_idx]
            det = high_dets[det_idx]
            track.update(det, self.frame_id)
            activated_tracks.append(track)

        unmatched_tracked = [tracked_pool[i] for i in u_tracked_1]
        remaining_high = [high_dets[i] for i in u_high_1]

        # --------------------------------------------------
        # Stage 2: unmatched tracked tracks vs low detections
        # --------------------------------------------------
        cost_2 = iou_distance(unmatched_tracked, low_dets, class_aware=self.class_aware)
        matches_2, u_tracked_2, _ = greedy_match(cost_2, self.low_match_thresh)

        for track_idx, det_idx in matches_2:
            track = unmatched_tracked[track_idx]
            det = low_dets[det_idx]
            track.update(det, self.frame_id)
            activated_tracks.append(track)

        for idx in u_tracked_2:
            track = unmatched_tracked[idx]
            track.mark_lost()
            newly_lost_tracks.append(track)

        # --------------------------------------------------
        # Stage 3: lost tracks vs remaining high detections
        # --------------------------------------------------
        cost_3 = iou_distance(self.lost_stracks, remaining_high, class_aware=self.class_aware)
        matches_3, u_lost_3, u_high_3 = greedy_match(cost_3, self.reactivate_match_thresh)

        for lost_idx, det_idx in matches_3:
            track = self.lost_stracks[lost_idx]
            det = remaining_high[det_idx]
            track.re_activate(det, self.frame_id, new_id=False)
            reactivated_tracks.append(track)

        # --------------------------------------------------
        # Stage 4: unmatched high detections start new tracks
        # --------------------------------------------------
        for det_idx in u_high_3:
            det = remaining_high[det_idx]
            if det.score >= self.new_track_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                activated_tracks.append(det)

        # --------------------------------------------------
        # Stage 5: age lost tracks out
        # --------------------------------------------------
        kept_lost_tracks = []
        for idx in u_lost_3:
            track = self.lost_stracks[idx]
            if (self.frame_id - track.frame_id) > self.buffer_size:
                track.mark_removed()
                newly_removed_tracks.append(track)
            else:
                kept_lost_tracks.append(track)

        kept_lost_tracks.extend(newly_lost_tracks)

        # --------------------------------------------------
        # Rebuild tracked list
        # --------------------------------------------------
        tracked_next = []

        for t in self.tracked_stracks:
            if t.state == TrackState.Tracked:
                tracked_next.append(t)

        tracked_next.extend(activated_tracks)
        tracked_next.extend(reactivated_tracks)

        tracked_next = self._remove_duplicate_ids_keep_first(tracked_next)

        # Remove anything that is now marked lost/removed
        tracked_next = [t for t in tracked_next if t.state == TrackState.Tracked]

        # Rebuild lost list
        tracked_ids = {t.track_id for t in tracked_next}
        lost_next = []

        for t in kept_lost_tracks:
            if t.state != TrackState.Lost:
                continue
            if t.track_id in tracked_ids:
                continue
            lost_next.append(t)

        lost_next = self._remove_duplicate_ids_keep_first(lost_next)

        self.tracked_stracks = tracked_next
        self.lost_stracks = lost_next
        self.removed_stracks.extend(newly_removed_tracks)

        return [t for t in self.tracked_stracks if t.is_activated]
