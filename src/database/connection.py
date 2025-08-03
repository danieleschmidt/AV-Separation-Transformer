"""Database connection management for AV-Separation-Transformer."""

import os
import logging
from typing import Optional, Dict, Any, AsyncContextManager
from contextlib import asynccontextmanager
from pathlib import Path

try:
    import asyncpg
    import aiosqlite
    ASYNC_DB_AVAILABLE = True
except ImportError:
    ASYNC_DB_AVAILABLE = False
    
import sqlite3
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager supporting multiple backends.
    
    Supports SQLite for development and PostgreSQL for production.
    Provides both sync and async interfaces.
    
    Example:
        >>> db = DatabaseConnection('sqlite:///av_separation.db')
        >>> with db.get_connection() as conn:
        ...     result = conn.execute('SELECT * FROM models')
        
        >>> async with db.get_async_connection() as conn:
        ...     result = await conn.fetch('SELECT * FROM models')
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600
    ):
        self.database_url = database_url or self._get_default_url()
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        
        self._local = threading.local()
        self._pool = None
        self._initialized = False
        
        # Parse database URL
        self.db_type, self.db_path = self._parse_url(self.database_url)
        
        logger.info(f"Database connection initialized: {self.db_type}")
    
    def _get_default_url(self) -> str:
        """Get default database URL from environment or use SQLite."""
        return os.getenv('DATABASE_URL', 'sqlite:///av_separation.db')
    
    def _parse_url(self, url: str) -> tuple[str, str]:
        """Parse database URL to extract type and connection details."""
        if url.startswith('sqlite:///'):
            db_type = 'sqlite'
            db_path = url.replace('sqlite:///', '')
        elif url.startswith('postgresql://') or url.startswith('postgres://'):
            db_type = 'postgresql'
            db_path = url
        else:
            raise ValueError(f"Unsupported database URL: {url}")
        
        return db_type, db_path
    
    def initialize(self) -> None:
        """Initialize database connection and create tables."""
        if self._initialized:
            return
        
        if self.db_type == 'sqlite':
            self._init_sqlite()
        elif self.db_type == 'postgresql':
            self._init_postgresql()
        
        self._create_tables()
        self._initialized = True
        
        logger.info("Database initialized successfully")
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Test connection
        conn = sqlite3.connect(self.db_path)
        conn.close()
    
    def _init_postgresql(self) -> None:
        """Initialize PostgreSQL connection pool."""
        try:
            import psycopg2
            from psycopg2 import pool
            
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                1, self.pool_size, self.db_path
            )
            
        except ImportError:
            raise ImportError("psycopg2 required for PostgreSQL support")
    
    def _create_tables(self) -> None:
        """Create necessary database tables."""
        schema_sql = """
        -- Models table for storing model metadata
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(255) NOT NULL UNIQUE,
            version VARCHAR(50) NOT NULL,
            model_type VARCHAR(100) NOT NULL,
            file_path TEXT NOT NULL,
            checksum VARCHAR(64),
            num_speakers INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON
        );
        
        -- Separation sessions for tracking processing jobs
        CREATE TABLE IF NOT EXISTS separation_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id VARCHAR(100) NOT NULL UNIQUE,
            model_id INTEGER,
            status VARCHAR(50) DEFAULT 'pending',
            input_audio_path TEXT,
            input_video_path TEXT,
            output_dir TEXT,
            num_speakers INTEGER,
            processing_time REAL,
            quality_metrics JSON,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models (id)
        );
        
        -- Performance metrics table
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id VARCHAR(100),
            metric_name VARCHAR(100) NOT NULL,
            metric_value REAL NOT NULL,
            metric_unit VARCHAR(50),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON,
            FOREIGN KEY (session_id) REFERENCES separation_sessions (session_id)
        );
        
        -- Model cache table for tracking cached models
        CREATE TABLE IF NOT EXISTS model_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key VARCHAR(255) NOT NULL UNIQUE,
            model_data BLOB,
            expiry_time TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- System health metrics
        CREATE TABLE IF NOT EXISTS system_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            component VARCHAR(100) NOT NULL,
            status VARCHAR(50) NOT NULL,
            cpu_usage REAL,
            memory_usage REAL,
            gpu_usage REAL,
            gpu_memory_usage REAL,
            disk_usage REAL,
            network_io JSON,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_models_name ON models (name);
        CREATE INDEX IF NOT EXISTS idx_sessions_status ON separation_sessions (status);
        CREATE INDEX IF NOT EXISTS idx_sessions_created ON separation_sessions (created_at);
        CREATE INDEX IF NOT EXISTS idx_metrics_session ON performance_metrics (session_id);
        CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics (metric_name);
        CREATE INDEX IF NOT EXISTS idx_cache_key ON model_cache (cache_key);
        CREATE INDEX IF NOT EXISTS idx_health_component ON system_health (component);
        CREATE INDEX IF NOT EXISTS idx_health_timestamp ON system_health (timestamp);
        """
        
        with self.get_connection() as conn:
            if self.db_type == 'sqlite':
                for statement in schema_sql.split(';'):
                    if statement.strip():
                        conn.execute(statement)
                conn.commit()
            elif self.db_type == 'postgresql':
                # Adapt SQL for PostgreSQL
                pg_schema = schema_sql.replace(
                    'INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY'
                ).replace('JSON', 'JSONB')
                
                with conn.cursor() as cursor:
                    cursor.execute(pg_schema)
                conn.commit()
    
    def get_connection(self):
        """Get database connection (sync)."""
        if self.db_type == 'sqlite':
            return self._get_sqlite_connection()
        elif self.db_type == 'postgresql':
            return self._get_postgresql_connection()
    
    def _get_sqlite_connection(self):
        """Get SQLite connection with proper configuration."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.pool_timeout,
            check_same_thread=False
        )
        
        # Enable foreign keys and WAL mode
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous = NORMAL')
        conn.execute('PRAGMA cache_size = -64000')  # 64MB cache
        
        # Row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        return conn
    
    def _get_postgresql_connection(self):
        """Get PostgreSQL connection from pool."""
        if self._pool is None:
            raise RuntimeError("PostgreSQL pool not initialized")
        
        return self._pool.getconn()
    
    def return_connection(self, conn) -> None:
        """Return connection to pool (PostgreSQL only)."""
        if self.db_type == 'postgresql' and self._pool:
            self._pool.putconn(conn)
    
    @asynccontextmanager
    async def get_async_connection(self) -> AsyncContextManager:
        """Get async database connection."""
        if not ASYNC_DB_AVAILABLE:
            raise ImportError("Async database libraries not available")
        
        if self.db_type == 'sqlite':
            async with aiosqlite.connect(self.db_path) as conn:
                # Enable foreign keys
                await conn.execute('PRAGMA foreign_keys = ON')
                yield conn
        
        elif self.db_type == 'postgresql':
            conn = await asyncpg.connect(self.db_path)
            try:
                yield conn
            finally:
                await conn.close()
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[tuple] = None
    ) -> list[Dict[str, Any]]:
        """Execute SELECT query and return results."""
        with self.get_connection() as conn:
            if self.db_type == 'sqlite':
                cursor = conn.execute(query, params or ())
                return [dict(row) for row in cursor.fetchall()]
            
            elif self.db_type == 'postgresql':
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def execute_command(
        self, 
        command: str, 
        params: Optional[tuple] = None
    ) -> int:
        """Execute INSERT/UPDATE/DELETE command."""
        with self.get_connection() as conn:
            if self.db_type == 'sqlite':
                cursor = conn.execute(command, params or ())
                conn.commit()
                return cursor.rowcount
            
            elif self.db_type == 'postgresql':
                with conn.cursor() as cursor:
                    cursor.execute(command, params)
                    rowcount = cursor.rowcount
                    conn.commit()
                    return rowcount
    
    async def execute_query_async(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> list[Dict[str, Any]]:
        """Execute async SELECT query."""
        async with self.get_async_connection() as conn:
            if self.db_type == 'sqlite':
                cursor = await conn.execute(query, params or ())
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
            
            elif self.db_type == 'postgresql':
                rows = await conn.fetch(query, *(params or ()))
                return [dict(row) for row in rows]
    
    async def execute_command_async(
        self,
        command: str,
        params: Optional[tuple] = None
    ) -> int:
        """Execute async INSERT/UPDATE/DELETE command."""
        async with self.get_async_connection() as conn:
            if self.db_type == 'sqlite':
                cursor = await conn.execute(command, params or ())
                await conn.commit()
                return cursor.rowcount
            
            elif self.db_type == 'postgresql':
                result = await conn.execute(command, *(params or ()))
                # PostgreSQL returns status string like 'INSERT 0 1'
                return int(result.split()[-1]) if result else 0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            with self.get_connection() as conn:
                if self.db_type == 'sqlite':
                    conn.execute('SELECT 1')
                elif self.db_type == 'postgresql':
                    with conn.cursor() as cursor:
                        cursor.execute('SELECT 1')
            
            return {
                'status': 'healthy',
                'database_type': self.db_type,
                'connection_pool_size': self.pool_size if self.db_type == 'postgresql' else 1
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'database_type': self.db_type
            }
    
    def close(self) -> None:
        """Close database connections."""
        if self.db_type == 'postgresql' and self._pool:
            self._pool.closeall()
            
        logger.info("Database connections closed")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except:
            pass


@lru_cache(maxsize=1)
def get_db_connection() -> DatabaseConnection:
    """Get singleton database connection instance."""
    db = DatabaseConnection()
    db.initialize()
    return db


class TransactionManager:
    """Context manager for database transactions."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.conn = None
    
    def __enter__(self):
        self.conn = self.db.get_connection()
        if self.db.db_type == 'sqlite':
            self.conn.execute('BEGIN')
        elif self.db.db_type == 'postgresql':
            self.conn.autocommit = False
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            if self.db.db_type == 'postgresql':
                self.db.return_connection(self.conn)
            else:
                self.conn.close()