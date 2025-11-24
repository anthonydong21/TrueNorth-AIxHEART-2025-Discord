import sqlite3
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "rate_limits.db")

def init_db():
    """
    Initialize the SQLite database for rate limiting.
    Creates the table if it doesn't exist.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create table for request logs
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS request_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create index on ip_address and timestamp for faster queries
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_ip_timestamp 
        ON request_logs (ip_address, timestamp)
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Rate limit database initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def log_request(ip_address: str):
    """
    Log a request from an IP address.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO request_logs (ip_address) VALUES (?)', (ip_address,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log request for {ip_address}: {e}")

def get_request_count(ip_address: str, hours: int = 24) -> int:
    """
    Get the number of requests from an IP address in the last N hours.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Calculate cut-off time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
        SELECT COUNT(*) FROM request_logs 
        WHERE ip_address = ? AND timestamp > ?
        ''', (ip_address, cutoff_time))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.error(f"Failed to get request count for {ip_address}: {e}")
        return 0

def cleanup_old_logs(days: int = 7):
    """
    Remove logs older than N days to keep database size manageable.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        cursor.execute('DELETE FROM request_logs WHERE timestamp < ?', (cutoff_time,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        logger.info(f"Cleaned up {deleted_count} old request logs")
    except Exception as e:
        logger.error(f"Failed to cleanup old logs: {e}")

