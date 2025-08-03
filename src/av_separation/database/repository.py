from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from .models import Experiment, Model, Evaluation, Dataset, TrainingRun, InferenceLog
from .connection import get_db_connection


class BaseRepository:
    def __init__(self, session: Optional[Session] = None):
        self.db = get_db_connection()
        self.session = session
    
    def _get_session(self) -> Session:
        return self.session if self.session else self.db.get_session()


class ExperimentRepository(BaseRepository):
    def create(self, name: str, description: str = None, config: Dict = None) -> Experiment:
        with self.db.session_scope() as session:
            experiment = Experiment(
                name=name,
                description=description,
                config=config,
                status="created"
            )
            session.add(experiment)
            session.flush()
            return experiment
    
    def get(self, experiment_id: int) -> Optional[Experiment]:
        with self.db.session_scope() as session:
            return session.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    def get_by_name(self, name: str) -> Optional[Experiment]:
        with self.db.session_scope() as session:
            return session.query(Experiment).filter(Experiment.name == name).first()
    
    def list(self, limit: int = 100, offset: int = 0) -> List[Experiment]:
        with self.db.session_scope() as session:
            return session.query(Experiment)\
                .order_by(desc(Experiment.created_at))\
                .limit(limit)\
                .offset(offset)\
                .all()
    
    def update_status(self, experiment_id: int, status: str) -> bool:
        with self.db.session_scope() as session:
            experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if experiment:
                experiment.status = status
                if status == "running":
                    experiment.started_at = datetime.utcnow()
                elif status in ["completed", "failed"]:
                    experiment.completed_at = datetime.utcnow()
                return True
            return False
    
    def delete(self, experiment_id: int) -> bool:
        with self.db.session_scope() as session:
            experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if experiment:
                session.delete(experiment)
                return True
            return False


class ModelRepository(BaseRepository):
    def create(
        self,
        name: str,
        experiment_id: int,
        architecture: str,
        parameters: int,
        config: Dict = None
    ) -> Model:
        with self.db.session_scope() as session:
            model = Model(
                name=name,
                experiment_id=experiment_id,
                architecture=architecture,
                parameters=parameters,
                config=config
            )
            session.add(model)
            session.flush()
            return model
    
    def get(self, model_id: int) -> Optional[Model]:
        with self.db.session_scope() as session:
            return session.query(Model).filter(Model.id == model_id).first()
    
    def get_by_name(self, name: str) -> Optional[Model]:
        with self.db.session_scope() as session:
            return session.query(Model).filter(Model.name == name).first()
    
    def list_by_experiment(self, experiment_id: int) -> List[Model]:
        with self.db.session_scope() as session:
            return session.query(Model)\
                .filter(Model.experiment_id == experiment_id)\
                .order_by(desc(Model.created_at))\
                .all()
    
    def update_checkpoint(self, model_id: int, checkpoint_path: str) -> bool:
        with self.db.session_scope() as session:
            model = session.query(Model).filter(Model.id == model_id).first()
            if model:
                model.checkpoint_path = checkpoint_path
                model.updated_at = datetime.utcnow()
                return True
            return False
    
    def update_metrics(self, model_id: int, metrics: Dict) -> bool:
        with self.db.session_scope() as session:
            model = session.query(Model).filter(Model.id == model_id).first()
            if model:
                model.metrics = metrics
                model.updated_at = datetime.utcnow()
                return True
            return False


class EvaluationRepository(BaseRepository):
    def create(
        self,
        experiment_id: int,
        model_id: int,
        dataset_id: int,
        metrics: Dict[str, float]
    ) -> Evaluation:
        with self.db.session_scope() as session:
            evaluation = Evaluation(
                experiment_id=experiment_id,
                model_id=model_id,
                dataset_id=dataset_id,
                metrics=metrics,
                si_snr=metrics.get("si_snr"),
                sdr=metrics.get("sdr"),
                pesq=metrics.get("pesq"),
                stoi=metrics.get("stoi"),
                latency_ms=metrics.get("latency_ms"),
                rtf=metrics.get("rtf")
            )
            session.add(evaluation)
            session.flush()
            return evaluation
    
    def get(self, evaluation_id: int) -> Optional[Evaluation]:
        with self.db.session_scope() as session:
            return session.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    
    def list_by_model(self, model_id: int) -> List[Evaluation]:
        with self.db.session_scope() as session:
            return session.query(Evaluation)\
                .filter(Evaluation.model_id == model_id)\
                .order_by(desc(Evaluation.created_at))\
                .all()
    
    def list_by_experiment(self, experiment_id: int) -> List[Evaluation]:
        with self.db.session_scope() as session:
            return session.query(Evaluation)\
                .filter(Evaluation.experiment_id == experiment_id)\
                .order_by(desc(Evaluation.created_at))\
                .all()
    
    def get_best_by_metric(
        self,
        experiment_id: int,
        metric: str = "si_snr",
        ascending: bool = False
    ) -> Optional[Evaluation]:
        with self.db.session_scope() as session:
            query = session.query(Evaluation)\
                .filter(Evaluation.experiment_id == experiment_id)
            
            if hasattr(Evaluation, metric):
                order_col = getattr(Evaluation, metric)
                if ascending:
                    query = query.order_by(order_col)
                else:
                    query = query.order_by(desc(order_col))
            
            return query.first()


class DatasetRepository(BaseRepository):
    def create(
        self,
        name: str,
        path: str,
        split: str = "train",
        num_samples: int = 0,
        total_duration: float = 0,
        config: Dict = None
    ) -> Dataset:
        with self.db.session_scope() as session:
            dataset = Dataset(
                name=name,
                path=path,
                split=split,
                num_samples=num_samples,
                total_duration=total_duration,
                config=config
            )
            session.add(dataset)
            session.flush()
            return dataset
    
    def get(self, dataset_id: int) -> Optional[Dataset]:
        with self.db.session_scope() as session:
            return session.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    def get_by_name(self, name: str) -> Optional[Dataset]:
        with self.db.session_scope() as session:
            return session.query(Dataset).filter(Dataset.name == name).first()
    
    def list(self, split: Optional[str] = None) -> List[Dataset]:
        with self.db.session_scope() as session:
            query = session.query(Dataset)
            if split:
                query = query.filter(Dataset.split == split)
            return query.order_by(desc(Dataset.created_at)).all()


class TrainingRunRepository(BaseRepository):
    def log(
        self,
        experiment_id: int,
        model_id: int,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        metrics: Dict = None
    ) -> TrainingRun:
        with self.db.session_scope() as session:
            run = TrainingRun(
                experiment_id=experiment_id,
                model_id=model_id,
                epoch=epoch,
                step=step,
                loss=loss,
                learning_rate=learning_rate,
                metrics=metrics
            )
            session.add(run)
            session.flush()
            return run
    
    def get_history(
        self,
        model_id: int,
        limit: int = 1000
    ) -> List[TrainingRun]:
        with self.db.session_scope() as session:
            return session.query(TrainingRun)\
                .filter(TrainingRun.model_id == model_id)\
                .order_by(desc(TrainingRun.created_at))\
                .limit(limit)\
                .all()


class InferenceLogRepository(BaseRepository):
    def log(
        self,
        model_id: int,
        input_path: str,
        output_path: str = None,
        num_speakers: int = 2,
        latency_ms: float = 0,
        success: bool = True,
        error_message: str = None
    ) -> InferenceLog:
        with self.db.session_scope() as session:
            log = InferenceLog(
                model_id=model_id,
                input_path=input_path,
                output_path=output_path,
                num_speakers=num_speakers,
                latency_ms=latency_ms,
                success=success,
                error_message=error_message
            )
            session.add(log)
            session.flush()
            return log
    
    def get_stats(self, model_id: int) -> Dict[str, Any]:
        with self.db.session_scope() as session:
            logs = session.query(InferenceLog)\
                .filter(InferenceLog.model_id == model_id)\
                .all()
            
            if not logs:
                return {}
            
            latencies = [log.latency_ms for log in logs if log.latency_ms]
            success_count = sum(1 for log in logs if log.success)
            
            return {
                "total_inferences": len(logs),
                "success_rate": success_count / len(logs) if logs else 0,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0
            }