import sqlite3
from typing import Optional, Tuple, List, Dict, Set

class DatabaseManager:
    """
    doc
    """
    def __init__(self, db_name: str, confing_db: str) -> None:
        self.db_name = db_name
        self.config_db = confing_db
        
        self.conn = sqlite3.connect(self.db_name)
        self.conn_config = sqlite3.connect(self.config_db)

        for c in (self.conn, self.conn_config):
            c.execute("PRAGMA journal_mode=WAL;")
            c.execute("PRAGMA synchronous=NORMAL;")
            c.execute("PRAGMA temp_store=MEMORY;")
        
        self._create_table()
    
    def close_conn(self) -> None:
        if hasattr(self, 'conn'): self.conn.close()
        if hasattr(self, 'conn_config'): self.conn_config.close()


    def _create_table(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS opendetData (
                    ID TEXT PRIMARY KEY,
                    timeStamp TEXT NOT NULL,
                    licensePlate TEXT,
                    prob REAL,
                    lpWidth TEXT,
                    numLP TEXT,
                    color TEXT,
                    colorProb REAL,
                    model TEXT,
                    modelProb REAL,
                    cam TEXT,
                    isAlert INTEGER NOT NULL DEFAULT 0,
                    personID TEXT,
                    isPerson INTEGER NOT NULL DEFAULT 0,
                    CHECK (isAlert IN (0, 1)),
                    CHECK (isPerson IN (0, 1))
                )
            """)

    def modify(self, record) -> None:
        query = """
            UPDATE opendetData 
            SET timeStamp = ?, licensePlate = ?, prob = ?, lpWidth = ?, numLP = ?,
            color = ?, colorProb = ?, model = ?, modelProb = ?, isAlert = ?
            WHERE ID = ?
        """
        with self.conn:
            self.conn.execute(query, record)
        

    def insert(self, record: Tuple):
        query = """
            INSERT INTO opendetData (
                ID, timeStamp, licensePlate, prob, lpWidth, numLP,
                color, colorProb, model, modelProb, cam, isAlert
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.conn:
            self.conn.execute(query, record)


    def read_lp_alert(self) -> Optional[Set[str]]:
        # 4. Return a set for $O(1)$ fast lookups in Python memory
        cursor = self.conn_config.execute("SELECT wantedLP FROM alerts WHERE wantedLP IS NOT NULL")
        rows = cursor.fetchall()
        return set(row[0] for row in rows) if rows else None

    def read_region(self) -> Tuple[str]:
        with self.conn_config:
            cursor = self.conn.execute("""SELECT frontCamArea FROM setting""")
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

    def get_batch_record(self, global_ids: List[int]) -> Dict[int, Optional[Tuple]]:
        if not global_ids:
            return {}
        
        str_ids = [str(id_) for id_ in global_ids]
        placeholders = ','.join('?' for _ in str_ids)
        query = f"SELECT ID, timeStamp, licensePlate, prob, lpWidth, numLP, color, colorProb, model, modelProb FROM opendetData WHERE ID IN ({placeholders})"
        
        cursor = self.conn.execute(query, str_ids)
        return {row[0]: (row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]) for row in cursor.fetchall()}

    def batch_exists_id(self, global_ids: List[str]) -> Dict[int, bool]:
        if not global_ids:
            return {}

        str_ids = [str(id_) for id_ in global_ids]
        placeholders = ','.join('?' for _ in str_ids)
        query = f"SELECT ID FROM opendetData WHERE ID IN ({placeholders})"
        
        cursor = self.conn.execute(query, str_ids)
        existing_ids = {row[0] for row in cursor.fetchall()} 
        
        return {id_: str(id_) in existing_ids for id_ in global_ids}

    def get_last_id(self) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT ID FROM opendetData ORDER BY ROWID DESC LIMIT 1").fetchone()
            return row[0] if row else None

    def delete_record(self, id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM opendetData WHERE ID = ?", (id,))
