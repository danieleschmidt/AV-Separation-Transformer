from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    config = Column(JSON)
    status = Column(String(50), default="created")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    models = relationship("Model", back_populates="experiment", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="experiment", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    name = Column(String(255), unique=True, nullable=False)
    version = Column(String(50))
    architecture = Column(String(100))
    parameters = Column(Integer)
    checkpoint_path = Column(String(500))
    config = Column(JSON)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    experiment = relationship("Experiment", back_populates="models")
    evaluations = relationship("Evaluation", back_populates="model", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "version": self.version,
            "architecture": self.architecture,
            "parameters": self.parameters,
            "checkpoint_path": self.checkpoint_path,
            "config": self.config,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Evaluation(Base):
    __tablename__ = "evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    model_id = Column(Integer, ForeignKey("models.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    metrics = Column(JSON)
    si_snr = Column(Float)
    sdr = Column(Float)
    pesq = Column(Float)
    stoi = Column(Float)
    latency_ms = Column(Float)
    rtf = Column(Float)
    config = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    
    experiment = relationship("Experiment", back_populates="evaluations")
    model = relationship("Model", back_populates="evaluations")
    dataset = relationship("Dataset", back_populates="evaluations")
    
    def to_dict(self):
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "metrics": self.metrics,
            "si_snr": self.si_snr,
            "sdr": self.sdr,
            "pesq": self.pesq,
            "stoi": self.stoi,
            "latency_ms": self.latency_ms,
            "rtf": self.rtf,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    path = Column(String(500))
    split = Column(String(50))
    num_samples = Column(Integer)
    total_duration = Column(Float)
    config = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    
    evaluations = relationship("Evaluation", back_populates="dataset")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "split": self.split,
            "num_samples": self.num_samples,
            "total_duration": self.total_duration,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TrainingRun(Base):
    __tablename__ = "training_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    model_id = Column(Integer, ForeignKey("models.id"))
    epoch = Column(Integer)
    step = Column(Integer)
    loss = Column(Float)
    learning_rate = Column(Float)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=func.now())


class InferenceLog(Base):
    __tablename__ = "inference_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"))
    input_path = Column(String(500))
    output_path = Column(String(500))
    num_speakers = Column(Integer)
    latency_ms = Column(Float)
    success = Column(Boolean)
    error_message = Column(Text)
    created_at = Column(DateTime, default=func.now())