#!/usr/bin/env python3
"""
Test script for MacBook hardware optimization validation.

This script runs comprehensive validation tests for the MacBook training pipeline
to ensure all optimization components work correctly.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from macbook_optimization.hardware_validation import HardwareOptimizationValidator
from macbook_optimization.macbook_training_pipeline import MacBookTrainingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run MacBook hardware optimization validation."""
    logger.info("Starting MacBook hardware optimization validation")
    
    try:
        # Initialize training pipeline
        logger.info("Initializing MacBook training pipeline...")
        training_pipeline = MacBookTrainingPipeline(output_dir="validation_output")
        
        # Initialize validator
        logger.info("Initializing hardware optimization validator...")
        validator = HardwareOptimizationValidator(
            training_pipeline=training_pipeline,
            test_duration_seconds=30.0  # Shorter duration for quick validation
        )
        
        # Run comprehensive validation
        logger.info("Running comprehensive validation suite...")
        validation_results = validator.run_comprehensive_validation()
        
        # Generate and display report
        report = validator.generate_validation_report(validation_results)
        print("\n" + report)
        
        # Save report to file
        report_file = Path("validation_output") / "hardware_optimization_validation_report.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to {report_file}")
        
        # Run optional stress test if system passed basic validation
        if validation_results.system_ready:
            logger.info("System passed basic validation. Running stress test...")
            stress_test_result = validator.run_stress_test(duration_minutes=2.0)  # 2-minute stress test
            
            print(f"\nSTRESS TEST RESULT:")
            print(f"Status: {'PASS' if stress_test_result.passed else 'FAIL'}")
            print(f"Score: {stress_test_result.score:.1f}/100")
            if stress_test_result.warnings:
                print("Warnings:")
                for warning in stress_test_result.warnings:
                    print(f"  - {warning}")
            if stress_test_result.errors:
                print("Errors:")
                for error in stress_test_result.errors:
                    print(f"  - {error}")
        else:
            logger.warning("System failed basic validation. Skipping stress test.")
        
        # Display summary
        summary = validator.get_validation_summary()
        print(f"\nVALIDATION SUMMARY:")
        print(f"Overall Grade: {summary['latest_validation']['overall_grade']}")
        print(f"System Ready: {'YES' if summary['latest_validation']['system_ready'] else 'NO'}")
        print(f"Tests Passed: {summary['latest_validation']['passed_tests']}/{summary['latest_validation']['total_tests']}")
        
        # Cleanup
        if training_pipeline:
            training_pipeline.cleanup()
        
        # Exit with appropriate code
        if validation_results.system_ready:
            logger.info("Hardware optimization validation completed successfully!")
            return 0
        else:
            logger.error("Hardware optimization validation failed!")
            return 1
    
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)