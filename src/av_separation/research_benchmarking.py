"""
🔬 TERRAGON RESEARCH: Advanced Benchmarking & Validation Suite
Comprehensive research validation for novel architectures

RESEARCH OBJECTIVES:
1. Comparative analysis of attention mechanisms
2. Performance validation against baselines  
3. Statistical significance testing
4. Publication-ready benchmarking
5. Novel algorithm validation

Author: Terragon Autonomous SDLC System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import scipy.stats as stats
from collections import defaultdict

from .models.mamba_fusion import MambaAudioVisualFusion
from .models.attention_alternatives import (
    LiquidTimeConstantNetwork, 
    RetentiveNetwork, 
    LinearAttention, 
    HybridAttentionFusion
)


@dataclass 
class BenchmarkConfig:
    """Configuration for benchmarking experiments"""
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    d_model: int = 512
    num_heads: int = 8
    num_runs: int = 10
    warmup_runs: int = 3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16]
        if self.sequence_lengths is None:
            self.sequence_lengths = [128, 256, 512, 1024, 2048]


class ResearchBenchmark:
    """
    🔬 Advanced benchmarking suite for attention mechanism research
    """
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.device = torch.device(self.config.device)
        self.results = defaultdict(list)
        
        # Initialize models for comparison
        self.models = self._initialize_models()
        
        print(f"🔬 Research Benchmark initialized on {self.device}")
        print(f"📊 Testing configurations: {len(self.config.batch_sizes)} batch sizes × "
              f"{len(self.config.sequence_lengths)} sequence lengths")
    
    def _initialize_models(self) -> Dict[str, nn.Module]:
        """Initialize all models for comparison"""
        d_model = self.config.d_model
        num_heads = self.config.num_heads
        
        # Mock config for complex models
        class MockConfig:
            def __init__(self):
                self.model = type('obj', (object,), {
                    'd_model': d_model,
                    'mamba_layers': 6,
                    'max_speakers': 4
                })
                self.audio = type('obj', (object,), {'d_model': d_model})
                self.video = type('obj', (object,), {'d_model': d_model // 2})
        
        config = MockConfig()
        
        models = {
            'mamba_fusion': MambaAudioVisualFusion(config).to(self.device),
            'liquid_network': LiquidTimeConstantNetwork(d_model, d_model, d_model).to(self.device),
            'retentive_network': RetentiveNetwork(d_model, num_heads).to(self.device),
            'linear_attention': LinearAttention(d_model, num_heads).to(self.device),
            'hybrid_attention': HybridAttentionFusion(d_model, num_heads).to(self.device),
            'traditional_attention': nn.MultiheadAttention(d_model, num_heads, batch_first=True).to(self.device)
        }
        
        # Set all models to eval mode
        for model in models.values():
            model.eval()
        
        return models
    
    def benchmark_throughput(self) -> Dict[str, Any]:
        """
        Benchmark throughput (samples/second) for all models
        """
        print("🚀 Benchmarking Throughput Performance...")
        throughput_results = {}
        
        for model_name, model in self.models.items():
            print(f"   Testing {model_name}...")
            model_throughput = {}
            
            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.sequence_lengths:
                    # Generate test data
                    if model_name == 'mamba_fusion':
                        audio_data = torch.randn(batch_size, seq_len, self.config.d_model).to(self.device)
                        video_data = torch.randn(batch_size, seq_len, self.config.d_model // 2).to(self.device)
                        test_data = (audio_data, video_data)
                    else:
                        test_data = torch.randn(batch_size, seq_len, self.config.d_model).to(self.device)
                    
                    # Warmup
                    with torch.no_grad():
                        for _ in range(self.config.warmup_runs):
                            _ = self._forward_pass(model, model_name, test_data)
                    
                    # Benchmark
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for _ in range(self.config.num_runs):
                            _ = self._forward_pass(model, model_name, test_data)
                    
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    end_time = time.time()
                    
                    # Calculate throughput
                    total_time = end_time - start_time
                    total_samples = batch_size * self.config.num_runs
                    throughput = total_samples / total_time
                    
                    model_throughput[f'batch_{batch_size}_seq_{seq_len}'] = throughput
            
            throughput_results[model_name] = model_throughput
        
        return throughput_results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """
        Benchmark memory usage for all models
        """
        print("💾 Benchmarking Memory Usage...")
        memory_results = {}
        
        if self.device.type != 'cuda':
            print("   ⚠️  Memory benchmarking requires CUDA")
            return {}
        
        for model_name, model in self.models.items():
            print(f"   Testing {model_name}...")
            model_memory = {}
            
            for batch_size in [1, 8, 16]:  # Limited for memory testing
                for seq_len in [512, 1024]:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    
                    # Generate test data  
                    if model_name == 'mamba_fusion':
                        audio_data = torch.randn(batch_size, seq_len, self.config.d_model).to(self.device)
                        video_data = torch.randn(batch_size, seq_len, self.config.d_model // 2).to(self.device)
                        test_data = (audio_data, video_data)
                    else:
                        test_data = torch.randn(batch_size, seq_len, self.config.d_model).to(self.device)
                    
                    # Forward pass
                    with torch.no_grad():
                        _ = self._forward_pass(model, model_name, test_data)
                    
                    # Get memory usage
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                    model_memory[f'batch_{batch_size}_seq_{seq_len}'] = peak_memory
            
            memory_results[model_name] = model_memory
        
        return memory_results
    
    def benchmark_accuracy(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark accuracy on synthetic audio-visual separation task
        """
        print("🎯 Benchmarking Accuracy Performance...")
        accuracy_results = {}
        
        # Generate synthetic ground truth data
        batch_size = 4
        seq_len = 512
        
        for model_name, model in self.models.items():
            print(f"   Testing {model_name}...")
            
            # Skip non-fusion models for audio-visual tasks
            if model_name != 'mamba_fusion' and 'fusion' not in model_name:
                continue
            
            accuracies = []
            
            for _ in range(num_samples // batch_size):
                # Generate synthetic mixed audio and clean targets
                clean_audio1 = torch.randn(batch_size, seq_len, self.config.d_model).to(self.device)
                clean_audio2 = torch.randn(batch_size, seq_len, self.config.d_model).to(self.device)
                mixed_audio = clean_audio1 + clean_audio2
                
                video_data = torch.randn(batch_size, seq_len, self.config.d_model // 2).to(self.device)
                
                with torch.no_grad():
                    if model_name == 'mamba_fusion':
                        separated, _ = model(mixed_audio, video_data)
                    else:
                        continue  # Skip non-fusion models
                
                # Simple accuracy metric: correlation with clean signal
                correlation = F.cosine_similarity(separated, clean_audio1, dim=-1).mean()
                accuracies.append(correlation.item())
            
            if accuracies:
                accuracy_results[model_name] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'all_scores': accuracies
                }
        
        return accuracy_results
    
    def computational_complexity_analysis(self) -> Dict[str, Any]:
        """
        Theoretical and empirical computational complexity analysis
        """
        print("📊 Analyzing Computational Complexity...")
        complexity_results = {}
        
        sequence_lengths = [64, 128, 256, 512, 1024, 2048]
        
        for model_name, model in self.models.items():
            print(f"   Analyzing {model_name}...")
            
            # Theoretical complexity
            theoretical = self._get_theoretical_complexity(model_name, self.config.d_model)
            
            # Empirical timing
            empirical_times = []
            for seq_len in sequence_lengths:
                if model_name == 'mamba_fusion':
                    audio_data = torch.randn(1, seq_len, self.config.d_model).to(self.device)
                    video_data = torch.randn(1, seq_len, self.config.d_model // 2).to(self.device)
                    test_data = (audio_data, video_data)
                else:
                    test_data = torch.randn(1, seq_len, self.config.d_model).to(self.device)
                
                # Time the forward pass
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(10):  # Multiple runs for stability
                        _ = self._forward_pass(model, model_name, test_data)
                
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                empirical_times.append((seq_len, avg_time))
            
            complexity_results[model_name] = {
                'theoretical': theoretical,
                'empirical': empirical_times,
                'scaling_factor': self._compute_scaling_factor(empirical_times)
            }
        
        return complexity_results
    
    def statistical_significance_test(self, results1: List[float], results2: List[float], 
                                    model1_name: str, model2_name: str) -> Dict[str, Any]:
        """
        Perform statistical significance testing between two models
        """
        print(f"📈 Statistical significance test: {model1_name} vs {model2_name}")
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(results1, results2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(results1) + np.var(results2)) / 2)
        cohens_d = (np.mean(results1) - np.mean(results2)) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference in means
        diff = np.array(results1) - np.array(results2)
        diff_mean = np.mean(diff)
        diff_std = np.std(diff, ddof=1)
        n = len(diff)
        
        # 95% confidence interval
        t_critical = stats.t.ppf(0.975, n-1)
        margin_error = t_critical * (diff_std / np.sqrt(n))
        ci_lower = diff_mean - margin_error
        ci_upper = diff_mean + margin_error
        
        return {
            'models': f"{model1_name} vs {model2_name}",
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_effect_size(abs(cohens_d)),
            'confidence_interval_95': (ci_lower, ci_upper),
            'mean_difference': diff_mean,
            'interpretation': self._interpret_results(p_value, cohens_d)
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmarking suite
        """
        print("🔬 TERRAGON RESEARCH: Comprehensive Benchmark Suite")
        print("=" * 70)
        
        results = {
            'config': {
                'device': str(self.device),
                'batch_sizes': self.config.batch_sizes,
                'sequence_lengths': self.config.sequence_lengths,
                'd_model': self.config.d_model,
                'num_heads': self.config.num_heads,
                'num_runs': self.config.num_runs
            },
            'model_parameters': {},
            'throughput': {},
            'memory_usage': {},
            'accuracy': {},
            'complexity': {},
            'statistical_tests': {}
        }
        
        # Model parameter counts
        for name, model in self.models.items():
            results['model_parameters'][name] = sum(p.numel() for p in model.parameters())
        
        # Run benchmarks
        results['throughput'] = self.benchmark_throughput()
        results['memory_usage'] = self.benchmark_memory_usage()
        results['accuracy'] = self.benchmark_accuracy()
        results['complexity'] = self.computational_complexity_analysis()
        
        # Statistical significance tests (example between key models)
        if 'mamba_fusion' in results['accuracy'] and len(results['accuracy']) > 1:
            model_names = list(results['accuracy'].keys())
            if len(model_names) >= 2:
                acc1 = results['accuracy'][model_names[0]]['all_scores']
                acc2 = results['accuracy'][model_names[1]]['all_scores'] if len(model_names) > 1 else acc1
                
                results['statistical_tests']['accuracy_comparison'] = self.statistical_significance_test(
                    acc1, acc2, model_names[0], model_names[1] if len(model_names) > 1 else model_names[0]
                )
        
        return results
    
    def _forward_pass(self, model: nn.Module, model_name: str, test_data) -> torch.Tensor:
        """Perform forward pass based on model type"""
        if model_name == 'mamba_fusion':
            audio_data, video_data = test_data
            output, _ = model(audio_data, video_data)
            return output
        elif model_name == 'liquid_network':
            output, _ = model(test_data)
            return output
        elif model_name == 'retentive_network':
            output, _ = model(test_data)
            return output
        elif model_name == 'hybrid_attention':
            result = model(test_data)
            return result['output']
        elif model_name == 'traditional_attention':
            output, _ = model(test_data, test_data, test_data)
            return output
        else:
            return model(test_data)
    
    def _get_theoretical_complexity(self, model_name: str, d_model: int) -> str:
        """Get theoretical computational complexity"""
        complexity_map = {
            'mamba_fusion': f"O(L × {d_model} × 16) - Linear",
            'liquid_network': f"O(L × {d_model}²) - Quadratic in d_model, Linear in L",
            'retentive_network': f"O(L × {d_model}²) - Linear in L",
            'linear_attention': f"O(L × {d_model}²) - Linear",
            'hybrid_attention': f"O(L × {d_model}²) - Mixed complexity",
            'traditional_attention': f"O(L² × {d_model}) - Quadratic in L"
        }
        return complexity_map.get(model_name, "Unknown")
    
    def _compute_scaling_factor(self, empirical_times: List[Tuple[int, float]]) -> float:
        """Compute empirical scaling factor"""
        if len(empirical_times) < 2:
            return 1.0
        
        # Simple linear regression to find scaling
        seq_lens = [t[0] for t in empirical_times]
        times = [t[1] for t in empirical_times]
        
        # Log-log regression to find scaling factor
        log_lens = np.log(seq_lens)
        log_times = np.log(times)
        
        # Linear regression: log(time) = scaling_factor * log(length) + constant
        slope, _ = np.polyfit(log_lens, log_times, 1)
        return slope
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return "Small"
        elif cohens_d < 0.5:
            return "Medium"
        elif cohens_d < 0.8:
            return "Large"
        else:
            return "Very Large"
    
    def _interpret_results(self, p_value: float, cohens_d: float) -> str:
        """Provide interpretation of statistical results"""
        if p_value < 0.001:
            significance = "highly significant"
        elif p_value < 0.01:
            significance = "very significant"
        elif p_value < 0.05:
            significance = "significant"
        else:
            significance = "not significant"
        
        effect = self._interpret_effect_size(abs(cohens_d)).lower()
        
        return f"Difference is {significance} with {effect} effect size"
    
    def save_results(self, results: Dict[str, Any], filename: str = "research_benchmark_results.json"):
        """Save benchmark results to file"""
        output_path = Path(filename)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📊 Results saved to {output_path}")


if __name__ == "__main__":
    # Run comprehensive research benchmark
    print("🔬 TERRAGON RESEARCH: Advanced Benchmarking Suite")
    print("=" * 70)
    
    # Configure benchmark
    config = BenchmarkConfig(
        batch_sizes=[1, 4, 8],
        sequence_lengths=[128, 256, 512],
        d_model=256,  # Smaller for faster testing
        num_heads=8,
        num_runs=5
    )
    
    # Run benchmark
    benchmark = ResearchBenchmark(config)
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    benchmark.save_results(results, "terragon_research_benchmark.json")
    
    # Print summary
    print("\n🏆 BENCHMARK SUMMARY")
    print("=" * 30)
    
    print("\n📊 Model Parameters:")
    for name, params in results['model_parameters'].items():
        print(f"   {name}: {params:,} parameters")
    
    print("\n🚀 Peak Throughput (samples/sec):")
    for model_name, throughput_data in results['throughput'].items():
        if throughput_data:
            max_throughput = max(throughput_data.values())
            print(f"   {model_name}: {max_throughput:.2f} samples/sec")
    
    if results['accuracy']:
        print("\n🎯 Accuracy Results:")
        for model_name, acc_data in results['accuracy'].items():
            print(f"   {model_name}: {acc_data['mean_accuracy']:.4f} ± {acc_data['std_accuracy']:.4f}")
    
    print("\n✅ Research benchmark completed successfully!")