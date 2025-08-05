#!/usr/bin/env python3
"""
AV-Separation-Transformer CLI Interface
Audio-Visual Speech Separation for real-time applications
"""

import click
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import json
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from .separator import AVSeparator
from .config import SeparatorConfig
from .version import __version__

console = Console()


@click.group()
@click.version_option(__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx, verbose):
    """AV-Separation-Transformer: Audio-Visual Speech Separation"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), 
              help='Output directory for separated audio files')
@click.option('--num-speakers', '-n', type=int, default=2,
              help='Number of speakers to separate')
@click.option('--device', '-d', type=str, default=None,
              help='Device to use (cuda/cpu)')
@click.option('--checkpoint', '-c', type=click.Path(exists=True),
              help='Path to model checkpoint')
@click.option('--config', type=click.Path(exists=True),
              help='Path to config file')
@click.option('--save-video', is_flag=True,
              help='Save processed video with face detection')
@click.pass_context
def separate(ctx, input_path, output_dir, num_speakers, device, 
            checkpoint, config, save_video):
    """Separate speakers from audio-visual input"""
    
    console.print(Panel.fit(
        f"[bold blue]AV-Separation-Transformer v{__version__}[/bold blue]\n"
        f"Separating {num_speakers} speakers from: {input_path}",
        title="Audio-Visual Speech Separation"
    ))
    
    try:
        # Load configuration
        if config:
            with open(config, 'r') as f:
                config_dict = json.load(f)
            separator_config = SeparatorConfig.from_dict(config_dict)
        else:
            separator_config = SeparatorConfig()
        
        # Initialize separator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Loading model...", total=None)
            
            separator = AVSeparator(
                num_speakers=num_speakers,
                device=device,
                checkpoint=checkpoint,
                config=separator_config
            )
            
            progress.update(task, description="Processing audio-visual input...")
            
            # Perform separation
            start_time = time.time()
            separated_audio = separator.separate(
                input_path=input_path,
                output_dir=output_dir,
                save_video=save_video
            )
            processing_time = time.time() - start_time
            
            progress.update(task, description="✓ Separation complete!", completed=True)
        
        # Display results
        results_table = Table(title="Separation Results")
        results_table.add_column("Speaker", style="cyan")
        results_table.add_column("Duration (s)", style="green")
        results_table.add_column("RMS Energy", style="yellow")
        
        for i, audio in enumerate(separated_audio):
            duration = len(audio) / separator_config.audio.sample_rate
            rms_energy = np.sqrt(np.mean(audio**2))
            results_table.add_row(
                f"Speaker {i+1}",
                f"{duration:.2f}",
                f"{rms_energy:.4f}"
            )
        
        console.print(results_table)
        console.print(f"\n[green]✓ Processing completed in {processing_time:.2f}s[/green]")
        
        if output_dir:
            console.print(f"[blue]Output saved to: {output_dir}[/blue]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if ctx.obj['verbose']:
            console.print_exception()
        return 1


@main.command()
@click.option('--model-path', '-m', type=click.Path(),
              help='Path to save the model')
@click.option('--format', '-f', type=click.Choice(['onnx', 'torchscript']),
              default='onnx', help='Export format')
@click.option('--opset-version', type=int, default=17,
              help='ONNX opset version')
@click.option('--optimize', is_flag=True,
              help='Apply optimizations')
@click.option('--quantize', is_flag=True,
              help='Apply INT8 quantization')
def export(model_path, format, opset_version, optimize, quantize):
    """Export model for deployment"""
    
    console.print(Panel.fit(
        f"[bold blue]Model Export[/bold blue]\n"
        f"Format: {format.upper()}\n"
        f"Optimizations: {'Enabled' if optimize else 'Disabled'}",
        title="Export Configuration"
    ))
    
    try:
        from .export import export_onnx, export_torchscript
        
        separator = AVSeparator()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Exporting model...", total=None)
            
            if format == 'onnx':
                export_onnx(
                    model=separator.model,
                    output_path=model_path or 'av_sepnet.onnx',
                    opset_version=opset_version,
                    optimize=optimize,
                    quantize=quantize
                )
            else:
                export_torchscript(
                    model=separator.model,
                    output_path=model_path or 'av_sepnet.pt',
                    optimize=optimize
                )
                
            progress.update(task, description="✓ Export complete!", completed=True)
        
        console.print(f"[green]✓ Model exported successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        return 1


@main.command()
@click.option('--model', '-m', type=str, default='av_sepnet_base',
              help='Model variant to benchmark')
@click.option('--device', '-d', type=str, default=None,
              help='Device to use for benchmarking')
@click.option('--iterations', '-i', type=int, default=100,
              help='Number of benchmark iterations')
@click.option('--batch-size', '-b', type=int, default=1,
              help='Batch size for benchmarking')
def benchmark(model, device, iterations, batch_size):
    """Benchmark separation performance"""
    
    console.print(Panel.fit(
        f"[bold blue]Performance Benchmark[/bold blue]\n"
        f"Model: {model}\n"
        f"Device: {device or 'auto'}\n"
        f"Iterations: {iterations}",
        title="Benchmark Configuration"
    ))
    
    try:
        separator = AVSeparator(device=device)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Running benchmark...", total=None)
            
            results = separator.benchmark(num_iterations=iterations)
            
            progress.update(task, description="✓ Benchmark complete!", completed=True)
        
        # Display benchmark results
        benchmark_table = Table(title="Benchmark Results")
        benchmark_table.add_column("Metric", style="cyan")
        benchmark_table.add_column("Value", style="green")
        
        for metric, value in results.items():
            if 'latency' in metric:
                benchmark_table.add_row(metric.replace('_', ' ').title(), f"{value:.2f} ms")
            elif metric == 'rtf':
                benchmark_table.add_row("Real-Time Factor", f"{value:.2f}x")
        
        console.print(benchmark_table)
        
        # Performance assessment
        rtf = results['rtf']
        if rtf < 1.0:
            console.print(f"[red]⚠ Real-time performance not achieved (RTF: {rtf:.2f})[/red]")
        else:
            console.print(f"[green]✓ Real-time capable (RTF: {rtf:.2f})[/green]")
            
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        return 1


@main.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--workers', default=1, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host, port, workers, reload):
    """Start the separation API server"""
    
    console.print(Panel.fit(
        f"[bold blue]AV-Separation API Server[/bold blue]\n"
        f"Host: {host}:{port}\n"
        f"Workers: {workers}",
        title="Server Configuration"
    ))
    
    try:
        import uvicorn
        from .api.app import app
        
        uvicorn.run(
            "av_separation.api.app:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload
        )
        
    except ImportError:
        console.print("[red]FastAPI and uvicorn required for API server[/red]")
        console.print("Install with: pip install 'av-separation-transformer[api]'")
        return 1
    except Exception as e:
        console.print(f"[red]Server failed to start: {e}[/red]")
        return 1


@main.command()
def info():
    """Display system and model information"""
    
    info_table = Table(title="System Information")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Status", style="green")
    
    # PyTorch info
    info_table.add_row("PyTorch Version", torch.__version__)
    info_table.add_row("CUDA Available", "✓" if torch.cuda.is_available() else "✗")
    
    if torch.cuda.is_available():
        info_table.add_row("CUDA Version", torch.version.cuda)
        info_table.add_row("GPU Count", str(torch.cuda.device_count()))
        info_table.add_row("GPU Name", torch.cuda.get_device_name(0))
    
    # Memory info
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        info_table.add_row("GPU Memory (Allocated)", f"{memory_allocated:.2f} GB")
        info_table.add_row("GPU Memory (Cached)", f"{memory_cached:.2f} GB")
    
    console.print(info_table)
    
    # Test model loading
    try:
        console.print("\n[blue]Testing model initialization...[/blue]")
        separator = AVSeparator()
        num_params = separator.model.get_num_params()
        console.print(f"[green]✓ Model loaded successfully ({num_params:,} parameters)[/green]")
    except Exception as e:
        console.print(f"[red]✗ Model loading failed: {e}[/red]")


if __name__ == '__main__':
    main()