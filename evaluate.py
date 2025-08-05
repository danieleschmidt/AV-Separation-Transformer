#!/usr/bin/env python3
"""
Evaluation Script for AV-Separation-Transformer
Comprehensive evaluation on various datasets and metrics
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.av_separation import AVSeparator, SeparatorConfig
from src.av_separation.utils.metrics import compute_all_metrics, SeparationMetrics


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class EvaluationDataset:
    """
    Dataset class for evaluation
    In practice, this would load real test data
    """
    
    def __init__(self, data_dir: str, config: SeparatorConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        
        # In practice, scan for test files
        # For demonstration, create synthetic test cases
        self.test_cases = []
        self.num_samples = 50  # Reduced for demonstration
        
        for i in range(self.num_samples):
            self.test_cases.append({
                'id': f'test_{i:03d}',
                'num_speakers': np.random.randint(2, 5),
                'duration': np.random.uniform(3.0, 8.0),
                'snr': np.random.uniform(-5, 20)  # Input SNR
            })
    
    def __len__(self):
        return len(self.test_cases)
    
    def __getitem__(self, idx):
        case = self.test_cases[idx]
        
        # Generate synthetic test data
        duration = case['duration']
        sample_rate = self.config.audio.sample_rate
        waveform_len = int(duration * sample_rate)
        
        # Create mixture and reference signals
        num_speakers = case['num_speakers']
        
        # Individual speaker signals
        clean_signals = []
        for s in range(num_speakers):
            # Create synthetic speech-like signal
            signal = np.random.randn(waveform_len) * 0.5
            
            # Add some periodicity to mimic speech
            freq = np.random.uniform(80, 300)  # Fundamental frequency
            t = np.linspace(0, duration, waveform_len)
            harmonic = 0.3 * np.sin(2 * np.pi * freq * t)
            signal += harmonic
            
            clean_signals.append(signal)
        
        clean_signals = np.array(clean_signals)
        
        # Create mixture with specified SNR
        mixture = np.sum(clean_signals, axis=0)
        
        # Add noise
        noise = np.random.randn(waveform_len) * 0.1
        mixture += noise
        
        # Normalize
        mixture = mixture / (np.max(np.abs(mixture)) + 1e-8) * 0.9
        clean_signals = clean_signals / (np.max(np.abs(clean_signals)) + 1e-8) * 0.9
        
        # Create corresponding video frames
        fps = self.config.video.fps
        num_frames = int(duration * fps)
        video_frames = np.random.randint(0, 256, 
                                       (num_frames, 3, *self.config.video.image_size),
                                       dtype=np.uint8)
        
        return {
            'id': case['id'],
            'mixture': mixture,
            'clean_signals': clean_signals,
            'video_frames': video_frames,
            'num_speakers': num_speakers,
            'duration': duration,
            'input_snr': case['snr']
        }


def evaluate_sample(
    separator: AVSeparator,
    sample: Dict[str, Any],
    compute_detailed_metrics: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a single sample
    
    Args:
        separator: AV separator model
        sample: Test sample data
        compute_detailed_metrics: Whether to compute detailed metrics
        
    Returns:
        Dictionary with evaluation results
    """
    
    sample_id = sample['id']
    mixture = sample['mixture']
    clean_signals = sample['clean_signals']
    video_frames = sample['video_frames']
    num_speakers = sample['num_speakers']
    
    # Perform separation
    try:
        # Create temporary files for input
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            sf.write(tmp_audio.name, mixture, separator.config.audio.sample_rate)
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
                # In practice, create proper video file
                # For now, we'll use the separate_stream method
                
                # Take middle frame for demonstration
                mid_frame = video_frames[len(video_frames) // 2]
                
                separated = separator.separate_stream(mixture, mid_frame.transpose(1, 2, 0))
        
        # Clean up temp files
        Path(tmp_audio.name).unlink(missing_ok=True)
        Path(tmp_video.name).unlink(missing_ok=True)
        
    except Exception as e:
        return {
            'sample_id': sample_id,
            'error': str(e),
            'success': False
        }
    
    # Compute metrics for each separated source
    results = {
        'sample_id': sample_id,
        'num_speakers_true': num_speakers,
        'num_speakers_pred': len(separated),
        'success': True,
        'metrics': {}
    }
    
    # Align separated signals with clean signals using optimal assignment
    if len(separated) == num_speakers:
        # Simple assignment for demonstration
        # In practice, use Hungarian algorithm for optimal assignment
        for i in range(num_speakers):
            sep_signal = separated[i]
            clean_signal = clean_signals[i]
            
            # Ensure same length
            min_len = min(len(sep_signal), len(clean_signal))
            sep_signal = sep_signal[:min_len]
            clean_signal = clean_signal[:min_len]
            
            # Compute metrics
            if compute_detailed_metrics:
                metrics = compute_all_metrics(
                    sep_signal, clean_signal,
                    separator.config.audio.sample_rate
                )
            else:
                # Compute only basic metrics for speed
                from src.av_separation.utils.metrics import compute_si_snr, compute_sdr
                
                metrics = {
                    'si_snr': compute_si_snr(sep_signal, clean_signal),
                    'sdr': compute_sdr(sep_signal, clean_signal)
                }
            
            results['metrics'][f'speaker_{i+1}'] = metrics
        
        # Compute average metrics
        all_metrics = list(results['metrics'].values())
        if all_metrics:
            avg_metrics = {}
            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics if not np.isnan(m[metric_name])]
                if values:
                    avg_metrics[metric_name] = np.mean(values)
                else:
                    avg_metrics[metric_name] = np.nan
            
            results['metrics']['average'] = avg_metrics
    else:
        results['error'] = f"Speaker count mismatch: {len(separated)} vs {num_speakers}"
    
    return results


def run_evaluation(
    separator: AVSeparator,
    test_dataset: EvaluationDataset,
    output_dir: Path,
    compute_detailed_metrics: bool = True
) -> Dict[str, Any]:
    """
    Run complete evaluation on test dataset
    
    Args:
        separator: AV separator model
        test_dataset: Test dataset
        output_dir: Output directory for results
        compute_detailed_metrics: Whether to compute detailed metrics
        
    Returns:
        Evaluation summary
    """
    
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics accumulator
    metrics_accumulator = SeparationMetrics(
        sample_rate=separator.config.audio.sample_rate,
        compute_pesq=compute_detailed_metrics,
        compute_stoi=compute_detailed_metrics
    )
    
    all_results = []
    successful_samples = 0
    
    logger.info(f"Evaluating on {len(test_dataset)} samples...")
    
    # Evaluate each sample
    for i, sample in enumerate(tqdm(test_dataset, desc="Evaluating")):
        result = evaluate_sample(separator, sample, compute_detailed_metrics)
        all_results.append(result)
        
        if result['success'] and 'average' in result['metrics']:
            successful_samples += 1
            
            # Update metrics accumulator (simplified for demonstration)
            avg_metrics = result['metrics']['average']
            if 'si_snr' in avg_metrics and not np.isnan(avg_metrics['si_snr']):
                # Create dummy tensors for the accumulator
                dummy_pred = torch.zeros(1, 1000)
                dummy_target = torch.zeros(1, 1000)
                metrics_accumulator.update(dummy_pred, dummy_target)
        
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(test_dataset)} samples")
    
    # Compute overall statistics
    logger.info("Computing overall statistics...")
    
    # Collect all metrics
    metric_names = ['si_snr', 'sdr', 'pesq', 'stoi']
    if compute_detailed_metrics:
        metric_names.extend(['spectral_convergence', 'log_spectral_distance'])
    
    overall_stats = {}
    
    for metric_name in metric_names:
        values = []
        
        for result in all_results:
            if (result['success'] and 'average' in result['metrics'] and 
                metric_name in result['metrics']['average']):
                
                value = result['metrics']['average'][metric_name]
                if not np.isnan(value):
                    values.append(value)
        
        if values:
            overall_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        else:
            overall_stats[metric_name] = {
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0
            }
    
    # Create summary
    summary = {
        'total_samples': len(test_dataset),
        'successful_samples': successful_samples,
        'success_rate': successful_samples / len(test_dataset),
        'overall_metrics': overall_stats,
        'model_info': {
            'num_parameters': separator.model.get_num_params(),
            'device': str(separator.device),
            'config': separator.config.to_dict()
        }
    }
    
    # Save detailed results
    results_file = output_dir / 'detailed_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Detailed results saved to {results_file}")
    
    # Save summary
    summary_file = output_dir / 'evaluation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Summary saved to {summary_file}")
    
    # Create CSV report
    create_csv_report(all_results, output_dir / 'results.csv')
    
    return summary


def create_csv_report(results: List[Dict], output_path: Path):
    """Create CSV report from results"""
    
    rows = []
    
    for result in results:
        if not result['success']:
            continue
        
        row = {
            'sample_id': result['sample_id'],
            'num_speakers_true': result['num_speakers_true'],
            'num_speakers_pred': result['num_speakers_pred'],
        }
        
        # Add average metrics
        if 'average' in result['metrics']:
            for metric_name, value in result['metrics']['average'].items():
                row[f'avg_{metric_name}'] = value
        
        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"CSV report saved to {output_path}")


def print_summary(summary: Dict[str, Any]):
    """Print evaluation summary"""
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"Total samples: {summary['total_samples']}")
    print(f"Successful samples: {summary['successful_samples']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    
    print(f"\nModel info:")
    print(f"  Parameters: {summary['model_info']['num_parameters']:,}")
    print(f"  Device: {summary['model_info']['device']}")
    
    print(f"\nOverall metrics:")
    
    metrics = summary['overall_metrics']
    
    # SI-SNR
    if 'si_snr' in metrics and metrics['si_snr']['count'] > 0:
        si_snr = metrics['si_snr']
        print(f"  SI-SNR: {si_snr['mean']:.2f} ± {si_snr['std']:.2f} dB")
    
    # SDR
    if 'sdr' in metrics and metrics['sdr']['count'] > 0:
        sdr = metrics['sdr']
        print(f"  SDR: {sdr['mean']:.2f} ± {sdr['std']:.2f} dB")
    
    # PESQ
    if 'pesq' in metrics and metrics['pesq']['count'] > 0:
        pesq = metrics['pesq']
        print(f"  PESQ: {pesq['mean']:.3f} ± {pesq['std']:.3f}")
    
    # STOI
    if 'stoi' in metrics and metrics['stoi']['count'] > 0:
        stoi = metrics['stoi']
        print(f"  STOI: {stoi['mean']:.3f} ± {stoi['std']:.3f}")
    
    print("="*60)


def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description='Evaluate AV-Separation-Transformer')
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast evaluation (skip detailed metrics)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = SeparatorConfig.from_dict(config_dict)
    else:
        config = SeparatorConfig()
    
    # Create separator
    logger.info("Loading model...")
    separator = AVSeparator(
        num_speakers=config.model.max_speakers,
        device=args.device,
        checkpoint=args.checkpoint,
        config=config
    )
    
    logger.info(f"Model loaded with {separator.model.get_num_params():,} parameters")
    
    # Create test dataset
    logger.info("Loading test dataset...")
    test_dataset = EvaluationDataset(args.data_dir, config)
    logger.info(f"Test dataset loaded with {len(test_dataset)} samples")
    
    # Run evaluation
    output_dir = Path(args.output_dir)
    
    summary = run_evaluation(
        separator=separator,
        test_dataset=test_dataset,
        output_dir=output_dir,
        compute_detailed_metrics=not args.fast
    )
    
    # Print summary
    print_summary(summary)
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()