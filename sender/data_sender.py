import time
import os
import base64
import socket
import requests
import sqlite3

from pathlib import Path

DATA_DB_PATH = "/home/pi/car-detector/database/data.db"
CHECKPOINT_DB_PATH = "/home/pi/car-detector/database/checkpoint.db"


class DataSender:

    LP_GLOB_PATTERN  = '[0-9][0-9]_[A-Za-z]*_[0-9][0-9][0-9]_[0-9][0-9]'

    def __init__(self):
        self.data_db_conn = sqlite3.connect(DATA_DB_PATH)
        self.checkpoint_db_conn = sqlite3.connect(CHECKPOINT_DB_PATH)

        self.t_send_data = 30
        self.BATCH_SIZE_DATA = 20
        self.t_now = time.time()
        self.t_last = 0

        self.IMAGE_DIR = "/home/pi/car-detector/public/detected/"
        self.SERVER_URL = 'https://sit-optic-android-jetson.wingom.ir/upload'
        self.CAM_ID = "CAدوربین نانوپای"

        https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        http_proxy  = os.environ.get("HTTP_PROXY")  or os.environ.get("http_proxy")
        if https_proxy or http_proxy:
            self.proxies = {
                "http":  http_proxy  or https_proxy,
                "https": https_proxy or http_proxy,
            }
            print(f"[DBWRITER] Using proxy for outbound HTTPS: {self.proxies['https']}")
        else:
            self.proxies = None
        self._create_chekpoint_table()


    def _create_chekpoint_table(self):
        with self.checkpoint_db_conn:
            self.checkpoint_db_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sendChkpt (
                row_id INTEGER PRIMARY KEY CHECK (row_id = 1),
                last_timestamp TEXT NOT NULL DEFAULT '',
                last_uuid TEXT NOT NULL DEFAULT ''
            )
            """
        )
            self.checkpoint_db_conn.execute(
                "INSERT OR IGNORE INTO sendChkpt (row_id) VALUES (1)"
            )
    
    # TODO add more params reading from database
    def read_next_batch(self, last_ts: str, last_uuid: str):

        base_query = """
            SELECT ID, timeStamp, color, model, licensePlate, prob, prob, colorProb, modelProb
            FROM opendetData
            WHERE licensePlate GLOB ?
        """

        params = [self.LP_GLOB_PATTERN]

        if last_ts:
            condition = """
                AND ((timeStamp > ?) OR (timeStamp = ? AND ID > ?))
            """
            full_query = base_query + condition + """
                ORDER BY timeStamp ASC, ID ASC
                LIMIT ?
            """
            params.extend([last_ts, last_ts, last_uuid, self.BATCH_SIZE_DATA])
        else:
            full_query = base_query + """
                ORDER BY timeStamp ASC, ID ASC
                LIMIT ?
            """
            params.append(self.BATCH_SIZE_DATA)

        return self.data_db_conn.execute(full_query, params).fetchall()


    def check_network_connection(self) -> bool:
        try:
            if self.proxies:
                from urllib.parse import urlparse
                p = urlparse(self.proxies["https"])
                host = p.hostname or "127.0.0.1"
                port = p.port or 8080
                socket.create_connection((host, port), timeout=3)
            else:
                socket.create_connection(
                    ("sit-optic-android-jetson.wingom.ir", 443), timeout=3
                )
            return True
        except OSError:
            return False

    def image_to_base64(self, image_path: Path):
        if image_path.is_file():
            return base64.b64encode(image_path.read_bytes()).decode('utf-8')
        return None

    @staticmethod
    def _f(v):
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def send_data(self):
        row_check_db = self.checkpoint_db_conn.execute(
            "SELECT last_timestamp, last_uuid FROM sendChkpt WHERE row_id = 1"
        ).fetchone()
        last_ts, last_uuid = (
            row_check_db[0] if row_check_db else '',
            row_check_db[1] if row_check_db else ''
        )
        rows = self.read_next_batch(last_ts, last_uuid)
        if not rows:
            print("[SEND] No new records to send.")
            return
        print(f"[SEND] {len(rows)} records read from database. Preparing...")

        payload = []
        payload_keys = []

        for row in rows:
            if len(row) == 9:
                (uuid_str, ts, color, model, lp,
                 lp_prob, lp_chars_prob, color_prob, model_prob) = row
            else:
                uuid_str, ts, color, model, lp = row[:5]
                lp_prob = lp_chars_prob = color_prob = model_prob = 0.0

            if not color:
                color = "unknown"
            if not model:
                model = "unknown"

            lp_prob       = self._f(lp_prob)
            lp_chars_prob = self._f(lp_chars_prob)
            color_prob    = self._f(color_prob)
            model_prob    = self._f(model_prob)

            img_path = Path(self.IMAGE_DIR) / f"{uuid_str}.jpg"
            b64 = self.image_to_base64(img_path)
            if b64:
                print(f"  [READ] Record ready to send -> ID: {uuid_str} | LP: {lp} "
                      f"| Color: {color} | Model: {model}")
                payload.append({
                    "camera_id":        self.CAM_ID,
                    "date":             ts,
                    "car_color":        color,
                    "car_model":        model,
                    "LP":               lp,
                    "image":            b64,
                    "class_name":       "car",
                    "box":              [0, 0, 0, 0],
                    "score":            lp_prob,
                    "model_pred_score": model_prob,
                    "color_pred_score": color_prob,
                    "lp_pred_score":    [lp_prob],
                    "lp_detail_scores": [lp_chars_prob],
                    "top_pred_models":  {model: model_prob} if model else {},
                    "top_pred_colours": {color: color_prob} if color else {},
                })
                payload_keys.append((ts, uuid_str))
            else:
                print(f"  [SKIP] Image for ID: {uuid_str} not found, skipped.")
        if not payload:
            print("[SEND] No records with valid images found. Sending cancelled.")
            return

        BATCH_SIZE = 1
        total = len(payload)
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        success_count = 0
        fail_count = 0

        last_sent_ts = last_ts
        last_sent_uuid = last_uuid

        print(f"[SEND] Total {total} records will be sent in {total_batches} batches of {BATCH_SIZE}.")
        print(f"[SEND] Server URL: {self.SERVER_URL}")

        for batch_idx in range(0, total, BATCH_SIZE):
            batch = payload[batch_idx:batch_idx + BATCH_SIZE]
            batch_keys = payload_keys[batch_idx:batch_idx + BATCH_SIZE]
            current_batch_num = (batch_idx // BATCH_SIZE) + 1

            print("=" * 70)
            print(f"[BATCH {current_batch_num}/{total_batches}] Batch content (contains {len(batch)} records):")
            print("=" * 70)
            for idx, item in enumerate(batch, start=1):
                preview = {k: (v[:60] + "...[TRUNCATED]" if k == "image" and v else v)
                           for k, v in item.items()}
                print(f"  Record {idx}/{len(batch)}:")
                for key, value in preview.items():
                    print(f"    {key}: {value}")
                print("-" * 70)

            try:
                r = requests.post(
                    self.SERVER_URL,
                    json=batch,
                    headers={'Content-Type': 'application/json'},
                    timeout=10,
                    proxies=self.proxies,
                )
                r.raise_for_status()
                print(f"[BATCH {current_batch_num}/{total_batches}] Send successful! Response code: {r.status_code}")
                success_count += len(batch)

                last_sent_ts, last_sent_uuid = batch_keys[-1]
                with self.checkpoint_db_conn:
                    self.checkpoint_db_conn.execute(
                        """ UPDATE sendChkpt
                            SET last_timestamp = ?, last_uuid = ?
                            WHERE row_id = 1""",
                            (last_sent_ts, last_sent_uuid))

            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                body = ""
                try:
                    body = e.response.text[:1000] if e.response is not None else ""
                except Exception:
                    pass

                print(f"[BATCH {current_batch_num}/{total_batches}] Send failed: {e}")
                if body:
                    print(f"  Server response body: {body}")

                if status in (400, 422):
                    # Genuine per-record validation error (bad payload). Retrying
                    # won't help, so skip past this record to avoid getting stuck.
                    bad_ids = [k[1] for k in batch_keys]
                    print(f"  [SKIP] Server rejected as invalid (HTTP {status}) record(s): {bad_ids}. "
                          f"Advancing checkpoint past them.")
                    fail_count += len(batch)
                    last_sent_ts, last_sent_uuid = batch_keys[-1]
                    with self.checkpoint_db_conn:
                        self.checkpoint_db_conn.execute(
                            """ UPDATE sendChkpt
                                SET last_timestamp = ?, last_uuid = ?
                                WHERE row_id = 1""",
                                (last_sent_ts, last_sent_uuid))
                    continue
                else:
                    # Auth / routing / rate-limit (401,403,404,429,...) or 5xx server
                    # error: NOT the record's fault. Do NOT advance the checkpoint;
                    # stop and retry next cycle so nothing is silently skipped.
                    print(f"  [STOP] HTTP {status} (not a payload problem). "
                          f"Checkpoint NOT advanced; will retry next cycle.")
                    fail_count += len(batch)
                    break

            except requests.RequestException as e:
                print(f"[BATCH {current_batch_num}/{total_batches}] Network/transport error: {e}")
                fail_count += len(batch)
                break

        print("=" * 70)
        print(f"[SEND] Final summary: success = {success_count} | failed = {fail_count} | total = {total}")
        print("=" * 70)

        if success_count > 0:
            print(f"[SEND] Checkpoint updated to ts={last_sent_ts}, uuid={last_sent_uuid}")
        else:
            print("[SEND] No successful sends. Checkpoint unchanged.")


    def run(self):
        print("[DBWRITER] Process loop started.")
        while True:
            self.t_now = time.time()
            if self.t_now - self.t_last > self.t_send_data:
                self.t_last = self.t_now
                if self.check_network_connection():
                    try:
                        self.send_data()
                    except Exception as e:
                        print(f"[DBWRITER] send_data error: {e}")
                else:
                    print("[DBWRITER] Network unreachable, skipping send.")
                continue
            else:
                time.sleep(0.05)

if __name__ == "__main__":

    data_sender = DataSender()
    data_sender.run()
