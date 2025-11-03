#!/usr/bin/env python3
"""
MacBook Email Training Examples Runner

This script runs various training examples to demonstrate and test different
configurations and scenarios on MacBook hardware.

Requirements: 1.5, 2.1, 5.4
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class TrainingExampleRunner:
    """Runner for training examples and tests."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the example runner."""
        self.project_root = project_root
        self.output_dir = Path(output_dir) if output_dir else Path("example_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_script = self.project_root / "train_email_classifier_macbook.py"
        self.examples_dir = self.project_root / "examples" / "macbook_training"
        
        self.results = []
    
    def run_hardware_detection_example(self) -> Dict[str, Any]:
        """Run hardware detection example."""
        print("🖥️  Running hardware detection example...")
        
        try:
            result = subprocess.run([
                sys.executable, str(self.training_script), '--detect-hardware'
            ], capture_output=True, text=True, timeout=30)
            
            success = result.returncode == 0
            
            return {
                'name': 'hardware_detection',
                'success': success,
                'duration': 0,  # Quick operation
                'output': result.stdout if success else result.stderr,
                'error': None if success else result.stderr
            }
        
        except Exception as e:
            return {
                'name': 'hardware_detection',
                'success': False,
                'duration': 0,
                'output': '',
                'error': str(e)
            }
    
    def run_sample_dataset_creation_example(self) -> Dict[str, Any]:
        """Run sample dataset creation example."""
        print("📊 Running sample dataset creation example...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                start_time = time.time()
                
                result = subprocess.run([
                    sys.executable, str(self.training_script),
                    '--create-sample-dataset',
                    '--dataset-path', temp_dir
                ], capture_output=True, text=True, timeout=60)
                
                duration = time.time() - start_time
                success = result.returncode == 0
                
                # Verify created files
                if success:
                    expected_files = [
                        "train/dataset.json",
                        "test/dataset.json",
                        "categories.json",
                        "vocab.json"
                    ]
                    
                    missing_files = []
                    for file_path in expected_files:
                        if not (Path(temp_dir) / file_path).exists():
                            missing_files.append(file_path)
                    
                    if missing_files:
                        success = False
                        error_msg = f"Missing files: {missing_files}"
                    else:
                        error_msg = None
                else:
                    error_msg = result.stderr
                
                return {
                    'name': 'sample_dataset_creation',
                    'success': success,
                    'duration': duration,
                    'output': result.stdout if success else result.stderr,
                    'error': error_msg
                }
            
            except Exception as e:
                return {
                    'name': 'sample_dataset_creation',
                    'success': False,
                    'duration': time.time() - start_time if 'start_time' in locals() else 0,
                    'output': '',
                    'error': str(e)
                }
    
    def run_configuration_validation_example(self) -> Dict[str, Any]:
        """Run configuration validation example."""
        print("⚙️  Running configuration validation example...")
        
        try:
            start_time = time.time()
            
            # Test with 16GB MacBook configuration
            config_path = self.examples_dir / "configs" / "macbook_16gb" / "email_classification.yaml"
            validator_script = self.examples_dir / "config_validator.py"
            
            result = subprocess.run([
                sys.executable, str(validator_script),
                '--config', str(config_path),
                '--memory-gb', '16',
                '--cpu-cores', '8'
            ], capture_output=True, text=True, timeout=30)
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            return {
                'name': 'configuration_validation',
                'success': success,
                'duration': duration,
                'output': result.stdout if success else result.stderr,
                'error': None if success else result.stderr
            }
        
        except Exception as e:
            return {
                'name': 'configuration_validation',
                'success': False,
                'duration': time.time() - start_time if 'start_time' in locals() else 0,
                'output': '',
                'error': str(e)
            }
    
    def run_quick_training_example(self, config_type: str = "8gb") -> Dict[str, Any]:
        """Run a quick training example with minimal steps."""
        print(f"🚀 Running quick training example ({config_type} MacBook)...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                start_time = time.time()
                
                # Create sample dataset first
                dataset_dir = Path(temp_dir) / "sample_dataset"
                subprocess.run([
                    sys.executable, str(self.training_script),
                    '--create-sample-dataset',
                    '--dataset-path', str(dataset_dir)
                ], check=True, capture_output=True)
                
                # Run quick training (very few steps for testing)
                output_dir = Path(temp_dir) / "training_output"
                
                # Get configuration parameters based on MacBook type
                if config_type == "8gb":
                    batch_size = 2
                    max_steps = 10  # Very quick for testing
                elif config_type == "16gb":
                    batch_size = 4
                    max_steps = 20
                else:  # 32gb
                    batch_size = 8
                    max_steps = 30
                
                result = subprocess.run([
                    sys.executable, str(self.training_script),
                    '--train',
                    '--dataset-path', str(dataset_dir),
                    '--output-dir', str(output_dir),
                    '--batch-size', str(batch_size),
                    '--max-steps', str(max_steps),
                    '--strategy', 'single',
                    '--log-level', 'WARNING'  # Reduce output
                ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
                
                duration = time.time() - start_time
                success = result.returncode == 0
                
                # Check if model was created
                if success:
                    model_files = list(output_dir.glob("*_final_model.pt"))
                    if not model_files:
                        success = False
                        error_msg = "No model file created"
                    else:
                        error_msg = None
                else:
                    error_msg = result.stderr
                
                return {
                    'name': f'quick_training_{config_type}',
                    'success': success,
                    'duration': duration,
                    'output': result.stdout[-1000:] if result.stdout else '',  # Last 1000 chars
                    'error': error_msg,
                    'config_type': config_type,
                    'steps': max_steps
                }
            
            except subprocess.TimeoutExpired:
                return {
                    'name': f'quick_training_{config_type}',
                    'success': False,
                    'duration': time.time() - start_time,
                    'output': '',
                    'error': 'Training timed out',
                    'config_type': config_type
                }
            except Exception as e:
                return {
                    'name': f'quick_training_{config_type}',
                    'success': False,
                    'duration': time.time() - start_time if 'start_time' in locals() else 0,
                    'output': '',
                    'error': str(e),
                    'config_type': config_type
                }
    
    def run_hyperparameter_optimization_example(self) -> Dict[str, Any]:
        """Run a quick hyperparameter optimization example."""
        print("🔧 Running hyperparameter optimization example...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                start_time = time.time()
                
                # Create sample dataset
                dataset_dir = Path(temp_dir) / "sample_dataset"
                subprocess.run([
                    sys.executable, str(self.training_script),
                    '--create-sample-dataset',
                    '--dataset-path', str(dataset_dir)
                ], check=True, capture_output=True)
                
                # Run quick hyperparameter optimization
                output_dir = Path(temp_dir) / "optimization_output"
                
                result = subprocess.run([
                    sys.executable, str(self.training_script),
                    '--optimize',
                    '--dataset-path', str(dataset_dir),
                    '--output-dir', str(output_dir),
                    '--num-trials', '3',  # Very few trials for testing
                    '--max-steps-per-trial', '5',  # Very few steps per trial
                    '--optimization-strategy', 'random',
                    '--log-level', 'WARNING'
                ], capture_output=True, text=True, timeout=180)  # 3 minute timeout
                
                duration = time.time() - start_time
                success = result.returncode == 0
                
                # Check if optimization results were created
                if success:
                    results_file = output_dir / "optimization_results.json"
                    if not results_file.exists():
                        success = False
                        error_msg = "No optimization results file created"
                    else:
                        error_msg = None
                else:
                    error_msg = result.stderr
                
                return {
                    'name': 'hyperparameter_optimization',
                    'success': success,
                    'duration': duration,
                    'output': result.stdout[-1000:] if result.stdout else '',
                    'error': error_msg,
                    'trials': 3
                }
            
            except subprocess.TimeoutExpired:
                return {
                    'name': 'hyperparameter_optimization',
                    'success': False,
                    'duration': time.time() - start_time,
                    'output': '',
                    'error': 'Optimization timed out'
                }
            except Exception as e:
                return {
                    'name': 'hyperparameter_optimization',
                    'success': False,
                    'duration': time.time() - start_time if 'start_time' in locals() else 0,
                    'output': '',
                    'error': str(e)
                }
    
    def run_all_examples(self, include_training: bool = True) -> List[Dict[str, Any]]:
        """Run all examples and return results."""
        print("🧪 Running MacBook Email Training Examples")
        print("=" * 50)
        
        examples = [
            self.run_hardware_detection_example,
            self.run_sample_dataset_creation_example,
            self.run_configuration_validation_example
        ]
        
        if include_training:
            # Add training examples
            examples.extend([
                lambda: self.run_quick_training_example("8gb"),
                self.run_hyperparameter_optimization_example
            ])
        
        results = []
        
        for i, example_func in enumerate(examples, 1):
            print(f"\n[{i}/{len(examples)}] ", end="")
            
            try:
                result = example_func()
                results.append(result)
                
                if result['success']:
                    print(f"✅ {result['name']} ({result['duration']:.1f}s)")
                else:
                    print(f"❌ {result['name']} - {result['error']}")
            
            except Exception as e:
                print(f"❌ Example failed with exception: {e}")
                results.append({
                    'name': f'example_{i}',
                    'success': False,
                    'duration': 0,
                    'output': '',
                    'error': str(e)
                })
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive report of example results."""
        total_examples = len(results)
        successful_examples = sum(1 for r in results if r['success'])
        failed_examples = total_examples - successful_examples
        
        total_duration = sum(r['duration'] for r in results)
        
        report = {
            'summary': {
                'total_examples': total_examples,
                'successful': successful_examples,
                'failed': failed_examples,
                'success_rate': successful_examples / total_examples if total_examples > 0 else 0,
                'total_duration': total_duration
            },
            'results': results,
            'recommendations': []
        }
        
        # Add recommendations based on results
        if failed_examples > 0:
            report['recommendations'].append(
                "Some examples failed. Check the troubleshooting guide for common issues."
            )
        
        if any(r.get('error') and 'memory' in r['error'].lower() for r in results):
            report['recommendations'].append(
                "Memory-related issues detected. Consider using smaller batch sizes or enabling memory optimizations."
            )
        
        if any(r.get('error') and 'timeout' in r['error'].lower() for r in results):
            report['recommendations'].append(
                "Timeout issues detected. Training may be slower than expected on this hardware."
            )
        
        return report


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MacBook email training examples")
    parser.add_argument("--output-dir", type=str, default="example_outputs",
                       help="Output directory for example results")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training examples (faster)")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")
    parser.add_argument("--save-report", action="store_true",
                       help="Save detailed report to file")
    
    args = parser.parse_args()
    
    # Create runner
    runner = TrainingExampleRunner(args.output_dir)
    
    # Run examples
    results = runner.run_all_examples(include_training=not args.skip_training)
    
    # Generate report
    report = runner.generate_report(results)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Example Results Summary")
    print("=" * 50)
    
    summary = report['summary']
    print(f"Total examples: {summary['total_examples']}")
    print(f"Successful: {summary['successful']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Total duration: {summary['total_duration']:.1f}s")
    
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Output JSON if requested
    if args.json:
        print("\n" + json.dumps(report, indent=2))
    
    # Save report if requested
    if args.save_report:
        report_file = Path(args.output_dir) / "examples_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Print next steps
    if summary['success_rate'] == 1.0:
        print("\n🎉 All examples passed! Your setup is working correctly.")
        print("\nNext steps:")
        print("1. Try training with your own email dataset")
        print("2. Experiment with different configurations")
        print("3. Review the performance optimization guide")
    else:
        print(f"\n⚠️  {summary['failed']} examples failed. Please check the issues above.")
        print("\nTroubleshooting:")
        print("1. Review failed example outputs")
        print("2. Check the troubleshooting guide")
        print("3. Validate your setup with: python examples/macbook_training/scripts/validate_setup.py")
    
    # Exit with appropriate code
    sys.exit(0 if summary['success_rate'] == 1.0 else 1)


if __name__ == "__main__":
    main()