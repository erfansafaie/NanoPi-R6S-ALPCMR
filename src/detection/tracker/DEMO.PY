import numpy as np
from bot_sort import BoTSORT


def fake_model_output(frame_id):
    if frame_id == 1:
        boxes = np.array([
            [100, 100, 180, 220],
            [300, 120, 380, 250],
        ], dtype=np.float32)
        scores = np.array([0.92, 0.88], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int32)

    elif frame_id == 2:
        boxes = np.array([
            [105, 102, 185, 222],
            [305, 123, 384, 252],
        ], dtype=np.float32)
        scores = np.array([0.90, 0.87], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int32)

    elif frame_id == 3:
        boxes = np.array([
            [110, 104, 190, 224],
            [310, 126, 388, 255],
        ], dtype=np.float32)
        scores = np.array([0.89, 0.22], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int32)

    elif frame_id == 4:
        boxes = np.array([
            [115, 106, 195, 226],
        ], dtype=np.float32)
        scores = np.array([0.91], dtype=np.float32)
        classes = np.array([0], dtype=np.int32)

    elif frame_id == 5:
        boxes = np.array([
            [120, 108, 200, 228],
            [318, 128, 396, 258],
        ], dtype=np.float32)
        scores = np.array([0.90, 0.86], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int32)

    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.int32)

    return boxes, scores, classes


def main():
    tracker = BoTSORT(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        match_thresh=0.7,
        low_match_thresh=0.85,
        reactivate_match_thresh=0.75,
        track_buffer=30,
        frame_rate=30,
        class_aware=True,
    )

    for frame_id in range(1, 8):
        boxes, scores, classes = fake_model_output(frame_id)
        tracks = tracker.update(boxes, scores, classes)

        print(f"\nFrame {frame_id}")
        print("Detections:")
        for i in range(len(boxes)):
            print(
                f"  det{i}: box={boxes[i].tolist()} "
                f"score={float(scores[i]):.2f} cls={int(classes[i])}"
            )

        print("Active Tracks:")
        for t in tracks:
            x1, y1, x2, y2 = t.tlbr
            print(
                f"  ID={t.track_id} cls={t.cls} score={t.score:.2f} "
                f"box={[round(float(x1),1), round(float(y1),1), round(float(x2),1), round(float(y2),1)]} "
                f"state={t.state.name}"
            )


if __name__ == "__main__":
    main()
