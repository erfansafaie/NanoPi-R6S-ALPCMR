import sqlite3
from typing import Optional, Tuple, List, Dict

class DatabaseManager:
    """
    doc
    """
    def __init__(self, db_name: str, confing_db: str) -> None:
        self.db_name = db_name
        self.config_db = confing_db
        
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
        with self._connect_config_db() as conn:
            conn.execute("PRAGMA journal_mode=WAL")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_name, isolation_level=None)
        # conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
        return conn

    def _connect_config_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config_db, isolation_level=None)
        # conn.row_factory = sqlite3.Row
        return conn

    def create_table(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS opendetData (
                    ID TEXT PRIMARY KEY,
                    timeStamp TEXT NOT NULL,
                    detTimeStamp TEXT NOT NULL,
                    licensePlate TEXT,
                    numLP TEXT,
                    lpWidth TEXT,
                    personID TEXT,
                    prob TEXT,
                    color TEXT,
                    colorProb REAL,
                    model TEXT,
                    modelProb REAL,
                    isAlert INTEGER NOT NULL DEFAULT 0,
                    cam TEXT,
                    isPerson INTEGER NOT NULL DEFAULT 0,
                    CHECK (isAlert IN (0, 1)),
                    CHECK (isPerson IN (0, 1))
                )
            """)
            # Add index for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_id ON opendetData (ID)")

    def modify(self, id: str, timestamp: str, lp: str, num_lp: str,
                prob_lp: str, is_alert: bool, lp_width: str) -> None:
        # Use parameterized query to prevent SQL injection
        query = """
            UPDATE opendetData 
            SET timeStamp = ?, licensePlate = ?, numLP = ?, prob=?, isAlert = ?, lpWidth = ?
            WHERE ID = ?
        """
        with self._connect() as conn:
            conn.execute(query, (timestamp, lp, num_lp, prob_lp, int(is_alert), lp_width, id))
            conn.commit()
        

    def insert(self, id: str, timestamp: str, det_timestamp: str, lp: str,
               num_lp: str, prob: str, color: str, color_prob: float, model: str,
               model_prob: float, is_alert: bool, cam: str, is_person: bool, lp_width: str) -> None:
        """
        doc
        """
        query = """
            INSERT INTO opendetData (
                ID, timeStamp, detTimeStamp, licensePlate, numLP, prob, color, 
                colorProb, model, modelProb, isAlert, cam, isPerson, lpWidth
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(query, (
                id, timestamp, det_timestamp, lp, num_lp, prob, color,
                color_prob, model, model_prob, int(is_alert), cam,
                int(is_person), lp_width
            ))
            conn.commit()

    def read_lp_alert(self) -> Tuple[str]:
        with self._connect_config_db() as conn:
            cursor = conn.execute("SELECT wantedLP FROM alerts WHERE wantedLP IS NOT NULL")
            rows = cursor.fetchall()
            if rows:
                return (tuple(row[0] for row in rows))
            else:
                return None

    def read_region(self) -> Tuple[str]:
        with self._connect_config_db() as conn:
            cursor = conn.execute("""SELECT frontCamArea FROM setting""")
            rows = cursor.fetchall()
            if rows[0][0].split():
                try:
                    return tuple(round(float(x),2) for x in rows[0][0].split("_"))
                except ValueError:
                    return None
            else:
                return None

    def get_comp_record(self, id: str) -> Optional[Tuple[str]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT numLP, lpWidth FROM opendetData WHERE ID = ?", 
                (id,)
            ).fetchone()
            return tuple(row) if row else None

    def batch_get_comp_record(self, global_ids: List[int]) -> Dict[int, Optional[Tuple]]:
        """Retrieve multiple records from the records table in a single query.
        
        Args:
            global_ids: List of integer IDs to retrieve (converted to strings for database query).
        
        Returns:
            Dictionary mapping each integer ID to its record tuple or None if not found.
        """
        if not global_ids:  # Handle empty input
            return {}
        str_ids = tuple(str(id_) for id_ in global_ids)
        with self._connect() as conn:
            placeholders = ','.join('?' for _ in str_ids)
            query = f"SELECT ID, licensePlate, numLP, lpWidth, prob FROM opendetData WHERE ID IN ({placeholders})"
            cursor = conn.execute(query, str_ids)
            rows = cursor.fetchall()
            return {int(id_): (lp, numLP, lp_width, lpchar_prob) for id_, lp, numLP, lp_width, lpchar_prob in rows}

    def batch_exists_id(self, global_ids: List[int]) -> Dict[int, bool]:
        """Check existence of multiple IDs in the records table in a single query.
        
        Args:
            global_ids: List of integer IDs to check (converted to strings for database query).
        
        Returns:
            Dictionary mapping each integer ID to a boolean indicating whether it exists.
        """
        if not global_ids:  # Handle empty input
            return {}

        str_ids = [str(id_) for id_ in global_ids]

        with self._connect() as conn:
            # Create placeholders for SQL IN clause
            placeholders = ','.join('?' for _ in str_ids)
            query = f"SELECT ID FROM opendetData WHERE ID IN ({placeholders})"
            cursor = conn.execute(query, str_ids)
            existing_ids = {row[0] for row in cursor.fetchall()}
            return {id_: str(id_) in existing_ids for id_ in global_ids}

    def get_last_id(self) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT ID FROM opendetData ORDER BY ROWID DESC LIMIT 1").fetchone()
            return row[0] if row else None

    def delete_record(self, id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM opendetData WHERE ID = ?", (id,))
