import os
from contextlib import contextmanager
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import redis
from pathlib import Path


class DatabaseConnection:
    def __init__(
        self,
        database_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        echo: bool = False
    ):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "sqlite:///./av_separation.db"
        )
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL",
            "redis://localhost:6379"
        )
        
        if self.database_url.startswith("sqlite"):
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=echo
            )
        else:
            self.engine = create_engine(
                self.database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=echo
            )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self._redis_client = None
    
    @property
    def redis_client(self):
        if self._redis_client is None:
            try:
                self._redis_client = redis.from_url(
                    self.redis_url,
                    decode_responses=True
                )
                self._redis_client.ping()
            except:
                self._redis_client = None
        return self._redis_client
    
    def get_session(self) -> Session:
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self):
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def create_tables(self):
        from .models import Base
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        from .models import Base
        Base.metadata.drop_all(bind=self.engine)
    
    def close(self):
        self.engine.dispose()
        if self._redis_client:
            self._redis_client.close()


_db_connection = None


def get_db_connection() -> DatabaseConnection:
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection


def init_database():
    db = get_db_connection()
    db.create_tables()
    return db