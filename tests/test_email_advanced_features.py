"""
Tests for Advanced Email Classification Features

Tests ensemble training, comprehensive evaluation system, and model interpretability
features for email classification with MacBook optimization.
"""

import unittest
import tempfile
import shutil
import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    DataLoader = None
    TensorDataset = None
    np = None
    TORCH_AVAILABLE = False

try:
    from macbook_optimization.email_ensemble_training import (
        EmailEnsembleTrainer, EnsembleTrainingConfig, EnsembleModelConfig,
        EnsembleTrainingResult
    )
    from macbook_optimization.email_comprehensive_evaluation import (
        ComprehensiveEmailEvaluator, ComprehensiveEvaluationResult,
        CategoryPerformanceMetrics
    )
    from macbook_optimization.email_interpretability import (
        EmailInterpretabilityAnalyzer, AttentionVisualization,
        FeatureImportance, ReasoningCycleAnalysis
    )
    from macbook_optimization.email_trm_integration import MacBookEmailTRM
    from macbook_optimization.email_training_config import EmailTrainingConfig
    from models.email_tokenizer import EmailTokenizer
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False


class MockEmailTokenizer:
    """Mock email tokenizer for testing."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {f"token_{i}": i for i in range(vocab_size)}
        self.id_to_token = {i: f"token_{i}" for i in range(vocab_size)}
    
    def encode(self, text):
        # Simple mock encoding
        words = text.split()[:10]  # Limit to 10 tokens
        return [hash(word) % self.vocab_size for word in words]
    
    def decode(self, token_ids):
        return " ".join([self.id_to_token.get(tid, f"<unk_{tid}>") for tid in token_ids])


class MockMacBookEmailTRM:
    """Mock EmailTRM model for testing."""
    
    def __init__(self, vocab_size=1000, num_categories=10, **kwargs):
        self.vocab_size = vocab_size
        self.num_categories = num_categories
        self.device = "cpu"
        
    def eval(self):
        pass
    
    def to(self, device):
        self.device = device
        return self
    
    def forward(self, inputs, labels=None, return_all_cycles=False, **kwargs):
        batch_size = inputs.shape[0] if hasattr(inputs, 'shape') else len(inputs)
        
        # Mock logits
        logits = torch.randn(batch_size, self.num_categories) if TORCH_AVAILABLE else [[0.1] * self.num_categories] * batch_size
        
        outputs = {
            'logits': logits,
            'num_cycles': 3
        }
        
        if labels is not None:
            outputs['loss'] = torch.tensor(0.5) if TORCH_AVAILABLE else 0.5
        
        if return_all_cycles:
            # Mock cycle outputs
            all_logits = torch.randn(3, batch_size, self.num_categories) if TORCH_AVAILABLE else [[[0.1] * self.num_categories] * batch_size] * 3
            outputs['all_logits'] = all_logits
        
        return outputs
    
    def predict(self, inputs, return_confidence=False, **kwargs):
        batch_size = inputs.shape[0] if hasattr(inputs, 'shape') else len(inputs)
        predictions = torch.randint(0, self.num_categories, (batch_size,)) if TORCH_AVAILABLE else [0] * batch_size
        
        if return_confidence:
            confidences = torch.rand(batch_size) if TORCH_AVAILABLE else [0.8] * batch_size
            return predictions, confidences
        
        return predictions
    
    def parameters(self):
        # Mock parameters
        return [torch.randn(10, 10)] if TORCH_AVAILABLE else []
    
    def state_dict(self):
        return {'mock_param': torch.randn(10, 10)} if TORCH_AVAILABLE else {}
    
    def load_state_dict(self, state_dict):
        pass


def create_mock_dataloader(num_samples=100, seq_len=50, num_categories=10):
    """Create mock dataloader for testing."""
    if not TORCH_AVAILABLE:
        return None
    
    inputs = torch.randint(0, 1000, (num_samples, seq_len))
    labels = torch.randint(0, num_categories, (num_samples,))
    
    dataset = TensorDataset(inputs, labels)
    
    # Mock the dataloader format expected by the system
    class MockDataLoader:
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __iter__(self):
            for i in range(0, len(self.dataset), 8):  # Batch size 8
                batch_inputs = self.dataset.tensors[0][i:i+8]
                batch_labels = self.dataset.tensors[1][i:i+8]
                
                batch = {
                    'inputs': batch_inputs,
                    'labels': batch_labels,
                    'puzzle_identifiers': None
                }
                
                yield "test", batch, len(batch_inputs)
    
    return MockDataLoader(dataset)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
@unittest.skipUnless(ADVANCED_FEATURES_AVAILABLE, "Advanced features not available")
class TestEmailEnsembleTraining(unittest.TestCase):
    """Test email ensemble training system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "ensemble_output")
        
        # Create mock base config
        self.base_config = EmailTrainingConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            batch_size=4,
            learning_rate=1e-4,
            num_email_categories=10
        )
        
        # Create ensemble config
        self.ensemble_config = EnsembleTrainingConfig(
            num_models=3,
            ensemble_name="test_ensemble",
            diversity_strategy="config_variation",
            voting_strategy="soft"
        )
        
        # Create mock dataset path
        self.dataset_path = os.path.join(self.temp_dir, "mock_dataset.json")
        with open(self.dataset_path, 'w') as f:
            json.dump({"emails": []}, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ensemble_trainer_initialization(self):
        """Test ensemble trainer initialization."""
        trainer = EmailEnsembleTrainer(output_dir=self.output_dir)
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.output_dir, Path(self.output_dir))
        self.assertIsNotNone(trainer.base_orchestrator)
        self.assertEqual(len(trainer.ensemble_history), 0)
    
    def test_create_ensemble_model_configs(self):
        """Test creation of diverse ensemble model configurations."""
        trainer = EmailEnsembleTrainer(output_dir=self.output_dir)
        
        model_configs = trainer.create_ensemble_model_configs(
            self.base_config, self.ensemble_config
        )
        
        self.assertEqual(len(model_configs), self.ensemble_config.num_models)
        
        # Check that configs are different (diversity)
        learning_rates = [config.config.learning_rate for config in model_configs]
        self.assertGreater(len(set(learning_rates)), 1, "Learning rates should be diverse")
        
        # Check config structure
        for config in model_configs:
            self.assertIsInstance(config, EnsembleModelConfig)
            self.assertIsNotNone(config.model_id)
            self.assertIsNotNone(config.model_name)
            self.assertIsInstance(config.config, EmailTrainingConfig)
    
    def test_config_variations(self):
        """Test configuration variation strategies."""
        trainer = EmailEnsembleTrainer(output_dir=self.output_dir)
        
        # Test config variation
        model_config = EnsembleModelConfig(
            model_id="test_model",
            model_name="test",
            config=self.base_config
        )
        
        varied_config = trainer._apply_config_variations(model_config, 0, 3)
        
        self.assertIsInstance(varied_config, EnsembleModelConfig)
        self.assertIsNotNone(varied_config.learning_rate_multiplier)
        self.assertIsNotNone(varied_config.hidden_size_multiplier)
    
    def test_data_variations(self):
        """Test data variation strategies."""
        trainer = EmailEnsembleTrainer(output_dir=self.output_dir)
        
        model_config = EnsembleModelConfig(
            model_id="test_model",
            model_name="test",
            config=self.base_config
        )
        
        varied_config = trainer._apply_data_variations(model_config, 1, 3)
        
        self.assertIsInstance(varied_config, EnsembleModelConfig)
        self.assertIsNotNone(varied_config.data_augmentation_prob)
        self.assertIsNotNone(varied_config.seed)
    
    def test_architecture_variations(self):
        """Test architecture variation strategies."""
        trainer = EmailEnsembleTrainer(output_dir=self.output_dir)
        
        model_config = EnsembleModelConfig(
            model_id="test_model",
            model_name="test",
            config=self.base_config
        )
        
        varied_config = trainer._apply_architecture_variations(model_config, 2, 3)
        
        self.assertIsInstance(varied_config, EnsembleModelConfig)
        self.assertIsNotNone(varied_config.use_hierarchical_attention)
        self.assertIsNotNone(varied_config.subject_attention_weight)
        self.assertIsNotNone(varied_config.pooling_strategy)
    
    @patch('macbook_optimization.email_ensemble_training.MacBookEmailTRM')
    @patch('macbook_optimization.email_ensemble_training.torch.load')
    @patch('macbook_optimization.email_ensemble_training.torch.save')
    def test_train_ensemble_mock(self, mock_save, mock_load, mock_model_class):
        """Test ensemble training with mocked components."""
        # Mock the orchestrator's training method
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.final_accuracy = 0.92
        mock_result.model_path = "/mock/path/model.pt"
        mock_result.total_training_time = 100.0
        mock_result.errors = []
        
        mock_orchestrator.execute_training_pipeline.return_value = mock_result
        
        # Mock model loading
        mock_load.return_value = {
            'model_state_dict': {},
            'config': {},
            'tokenizer_vocab': {}
        }
        
        # Mock model creation
        mock_model = MockMacBookEmailTRM()
        mock_model_class.return_value = mock_model
        
        trainer = EmailEnsembleTrainer(
            output_dir=self.output_dir,
            base_orchestrator=mock_orchestrator
        )
        
        result = trainer.train_ensemble(
            dataset_path=self.dataset_path,
            base_config=self.base_config,
            ensemble_config=self.ensemble_config
        )
        
        self.assertIsInstance(result, EnsembleTrainingResult)
        self.assertTrue(result.success)
        self.assertEqual(len(result.model_configs), self.ensemble_config.num_models)
        self.assertGreater(result.best_individual_accuracy, 0.0)
    
    def test_ensemble_summary(self):
        """Test ensemble training summary generation."""
        trainer = EmailEnsembleTrainer(output_dir=self.output_dir)
        
        # Test empty summary
        summary = trainer.get_ensemble_summary()
        self.assertIn("message", summary)
        
        # Add mock result to history
        mock_result = EnsembleTrainingResult(
            success=True,
            ensemble_id="test_ensemble",
            start_time=torch.cuda.Event(enable_timing=True),
            end_time=None,
            config=self.ensemble_config,
            model_configs=[],
            individual_results=[],
            individual_accuracies=[0.90, 0.92, 0.88],
            individual_f1_scores=[0.89, 0.91, 0.87],
            ensemble_accuracy=0.94,
            ensemble_f1_macro=0.93,
            ensemble_f1_micro=0.94,
            best_individual_accuracy=0.92,
            ensemble_improvement=0.02,
            ensemble_model_path=None,
            individual_model_paths=[],
            total_training_time=300.0,
            average_individual_training_time=100.0,
            errors=[],
            warnings=[]
        )
        
        trainer.ensemble_history.append(mock_result)
        
        summary = trainer.get_ensemble_summary()
        self.assertEqual(summary["total_ensemble_runs"], 1)
        self.assertEqual(summary["successful_ensemble_runs"], 1)
        self.assertEqual(summary["best_ensemble_accuracy"], 0.94)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
@unittest.skipUnless(ADVANCED_FEATURES_AVAILABLE, "Advanced features not available")
class TestComprehensiveEvaluation(unittest.TestCase):
    """Test comprehensive evaluation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "evaluation_output")
        
        self.category_names = [
            "Newsletter", "Work", "Personal", "Spam", "Promotional",
            "Social", "Finance", "Travel", "Shopping", "Other"
        ]
        
        self.evaluator = ComprehensiveEmailEvaluator(
            category_names=self.category_names,
            output_dir=self.output_dir,
            confidence_bins=5,
            save_detailed_predictions=True
        )
        
        self.mock_model = MockMacBookEmailTRM(num_categories=len(self.category_names))
        self.mock_dataloader = create_mock_dataloader(
            num_samples=50, 
            seq_len=20, 
            num_categories=len(self.category_names)
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        self.assertIsNotNone(self.evaluator)
        self.assertEqual(self.evaluator.category_names, self.category_names)
        self.assertEqual(self.evaluator.num_categories, len(self.category_names))
        self.assertEqual(self.evaluator.confidence_bins, 5)
        self.assertTrue(self.evaluator.save_detailed_predictions)
    
    def test_evaluate_model(self):
        """Test comprehensive model evaluation."""
        result = self.evaluator.evaluate_model(
            model=self.mock_model,
            dataloader=self.mock_dataloader,
            device="cpu",
            enable_uncertainty_estimation=True
        )
        
        self.assertIsInstance(result, ComprehensiveEvaluationResult)
        self.assertIsNotNone(result.evaluation_id)
        self.assertGreater(result.total_samples, 0)
        self.assertEqual(result.num_categories, len(self.category_names))
        self.assertIsInstance(result.category_metrics, dict)
        self.assertIsInstance(result.confusion_matrix, list)
        self.assertIsInstance(result.reliability_diagram_data, list)
    
    def test_category_metrics_computation(self):
        """Test per-category metrics computation."""
        # Create mock data
        predictions = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        labels = np.array([0, 1, 2, 1, 1, 2, 0, 0, 2, 0])
        probabilities = np.random.rand(10, len(self.category_names))
        
        category_metrics = self.evaluator._compute_category_metrics(
            predictions, labels, probabilities
        )
        
        self.assertIsInstance(category_metrics, dict)
        
        # Check that we have metrics for categories that appear in the data
        for category_name in self.category_names[:3]:  # First 3 categories used
            if category_name in category_metrics:
                metrics = category_metrics[category_name]
                self.assertIsInstance(metrics, CategoryPerformanceMetrics)
                self.assertIsInstance(metrics.precision, float)
                self.assertIsInstance(metrics.recall, float)
                self.assertIsInstance(metrics.f1_score, float)
                self.assertGreaterEqual(metrics.precision, 0.0)
                self.assertLessEqual(metrics.precision, 1.0)
    
    def test_confidence_calibration_analysis(self):
        """Test confidence calibration analysis."""
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 1, 0, 2])
        probabilities = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.6, 0.3],
            [0.9, 0.05, 0.05],
            [0.3, 0.3, 0.4]
        ])
        
        # Pad probabilities to match category count
        full_probs = np.zeros((5, len(self.category_names)))
        full_probs[:, :3] = probabilities
        
        analysis = self.evaluator._analyze_confidence_calibration(
            predictions, labels, full_probs
        )
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('overall_confidence', analysis)
        self.assertIn('expected_calibration_error', analysis)
        self.assertIn('reliability_data', analysis)
        self.assertIn('histogram_data', analysis)
        
        self.assertGreaterEqual(analysis['overall_confidence'], 0.0)
        self.assertLessEqual(analysis['overall_confidence'], 1.0)
        self.assertGreaterEqual(analysis['expected_calibration_error'], 0.0)
    
    def test_uncertainty_estimation(self):
        """Test prediction uncertainty estimation."""
        # Mock model outputs with multiple cycles
        mock_outputs = []
        for _ in range(3):  # 3 batches
            output = {
                'probabilities': torch.rand(8, len(self.category_names)),
                'predictions': torch.randint(0, len(self.category_names), (8,)),
                'labels': torch.randint(0, len(self.category_names), (8,)),
                'all_logits': torch.randn(3, 8, len(self.category_names)),  # 3 cycles
                'num_cycles': 3
            }
            mock_outputs.append(output)
        
        self.evaluator.model_outputs = mock_outputs
        
        uncertainty_metrics = self.evaluator._estimate_prediction_uncertainty()
        
        self.assertIsInstance(uncertainty_metrics, dict)
        self.assertIn('prediction_entropy', uncertainty_metrics)
        self.assertIn('epistemic_uncertainty', uncertainty_metrics)
        self.assertIn('aleatoric_uncertainty', uncertainty_metrics)
        
        self.assertGreaterEqual(uncertainty_metrics['prediction_entropy'], 0.0)
        self.assertGreaterEqual(uncertainty_metrics['epistemic_uncertainty'], 0.0)
        self.assertGreaterEqual(uncertainty_metrics['aleatoric_uncertainty'], 0.0)
    
    def test_error_analysis(self):
        """Test prediction error analysis."""
        predictions = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        labels = np.array([0, 1, 1, 1, 1, 2, 0, 0, 2, 0])  # Some errors
        probabilities = np.random.rand(10, len(self.category_names))
        
        error_analysis = self.evaluator._analyze_prediction_errors(
            predictions, labels, probabilities
        )
        
        self.assertIsInstance(error_analysis, dict)
        self.assertIn('error_patterns', error_analysis)
        self.assertIn('difficult_samples', error_analysis)
        
        self.assertIsInstance(error_analysis['error_patterns'], dict)
        self.assertIsInstance(error_analysis['difficult_samples'], list)
    
    def test_performance_by_confidence(self):
        """Test performance analysis by confidence bins."""
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 1, 0, 2])
        probabilities = np.array([
            [0.9, 0.05, 0.05],  # High confidence, correct
            [0.2, 0.7, 0.1],    # Medium confidence, correct
            [0.1, 0.6, 0.3],    # Medium confidence, incorrect
            [0.8, 0.1, 0.1],    # High confidence, correct
            [0.3, 0.3, 0.4]     # Low confidence, incorrect
        ])
        
        # Pad probabilities
        full_probs = np.zeros((5, len(self.category_names)))
        full_probs[:, :3] = probabilities
        
        performance_analysis = self.evaluator._analyze_performance_by_confidence(
            predictions, labels, full_probs
        )
        
        self.assertIsInstance(performance_analysis, list)
        
        for bin_data in performance_analysis:
            self.assertIn('bin_lower', bin_data)
            self.assertIn('bin_upper', bin_data)
            self.assertIn('accuracy', bin_data)
            self.assertIn('count', bin_data)
            self.assertGreaterEqual(bin_data['accuracy'], 0.0)
            self.assertLessEqual(bin_data['accuracy'], 1.0)
    
    def test_timing_analysis(self):
        """Test timing performance analysis."""
        # Mock timing data
        self.evaluator.inference_times = [0.01, 0.015, 0.012, 0.008, 0.02]
        
        timing_analysis = self.evaluator._analyze_timing_performance()
        
        self.assertIsInstance(timing_analysis, dict)
        self.assertIn('inference_time_stats', timing_analysis)
        self.assertIn('throughput_metrics', timing_analysis)
        
        stats = timing_analysis['inference_time_stats']
        self.assertIn('mean_time_per_sample', stats)
        self.assertIn('std_time_per_sample', stats)
        self.assertIn('p95_time_per_sample', stats)
        
        throughput = timing_analysis['throughput_metrics']
        self.assertIn('samples_per_second', throughput)
        self.assertIn('total_samples', throughput)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
@unittest.skipUnless(ADVANCED_FEATURES_AVAILABLE, "Advanced features not available")
class TestEmailInterpretability(unittest.TestCase):
    """Test email model interpretability features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "interpretability_output")
        
        self.tokenizer = MockEmailTokenizer(vocab_size=1000)
        self.category_names = [
            "Newsletter", "Work", "Personal", "Spam", "Promotional",
            "Social", "Finance", "Travel", "Shopping", "Other"
        ]
        
        self.analyzer = EmailInterpretabilityAnalyzer(
            tokenizer=self.tokenizer,
            category_names=self.category_names,
            output_dir=self.output_dir
        )
        
        self.mock_model = MockMacBookEmailTRM(num_categories=len(self.category_names))
        
        # Create sample input
        self.sample_input = torch.randint(0, 1000, (1, 20))
        self.sample_label = torch.randint(0, len(self.category_names), (1,))
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyzer_initialization(self):
        """Test interpretability analyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.category_names, self.category_names)
        self.assertIsInstance(self.analyzer.tokenizer, MockEmailTokenizer)
        self.assertEqual(len(self.analyzer.attention_visualizations), 0)
        self.assertEqual(len(self.analyzer.reasoning_analyses), 0)
    
    def test_analyze_sample_attention(self):
        """Test attention analysis for a single sample."""
        visualization = self.analyzer.analyze_sample_attention(
            model=self.mock_model,
            inputs=self.sample_input,
            labels=self.sample_label,
            email_text="Test email content",
            sample_id="test_sample_001"
        )
        
        self.assertIsInstance(visualization, AttentionVisualization)
        self.assertEqual(visualization.sample_id, "test_sample_001")
        self.assertEqual(visualization.email_text, "Test email content")
        self.assertIsInstance(visualization.tokens, list)
        self.assertIsInstance(visualization.attention_weights, list)
        self.assertIsInstance(visualization.structure_attention, dict)
        self.assertIn(visualization.predicted_category, self.category_names)
        self.assertGreaterEqual(visualization.confidence, 0.0)
        self.assertLessEqual(visualization.confidence, 1.0)
    
    def test_analyze_reasoning_cycles(self):
        """Test reasoning cycle analysis."""
        analysis = self.analyzer.analyze_reasoning_cycles(
            model=self.mock_model,
            inputs=self.sample_input,
            labels=self.sample_label,
            sample_id="reasoning_test_001"
        )
        
        self.assertIsInstance(analysis, ReasoningCycleAnalysis)
        self.assertEqual(analysis.sample_id, "reasoning_test_001")
        self.assertGreater(analysis.num_cycles, 0)
        self.assertIsInstance(analysis.cycle_predictions, list)
        self.assertIsInstance(analysis.cycle_confidences, list)
        self.assertIsInstance(analysis.cycle_entropies, list)
        self.assertGreaterEqual(analysis.prediction_stability, 0.0)
        self.assertLessEqual(analysis.prediction_stability, 1.0)
        self.assertGreaterEqual(analysis.attention_stability, 0.0)
        self.assertLessEqual(analysis.attention_stability, 1.0)
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis."""
        # Create mock dataloader
        mock_dataloader = create_mock_dataloader(num_samples=20, seq_len=15)
        
        if mock_dataloader is None:
            self.skipTest("Mock dataloader creation failed")
        
        # Test gradient-based importance
        importance = self.analyzer.analyze_feature_importance(
            model=self.mock_model,
            dataloader=mock_dataloader,
            num_samples=10,
            method="gradient"
        )
        
        self.assertIsInstance(importance, FeatureImportance)
        self.assertIsInstance(importance.token_importance, dict)
        self.assertIsInstance(importance.structure_importance, dict)
        self.assertIsInstance(importance.category_token_importance, dict)
        self.assertIsInstance(importance.most_important_tokens, list)
        self.assertIsInstance(importance.least_important_tokens, list)
    
    def test_gradient_importance_computation(self):
        """Test gradient-based importance computation."""
        importance_scores = self.analyzer._compute_gradient_importance(
            model=self.mock_model,
            inputs=self.sample_input,
            labels=self.sample_label
        )
        
        self.assertIsInstance(importance_scores, list)
        self.assertEqual(len(importance_scores), self.sample_input.shape[1])
        
        for score in importance_scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
    
    def test_permutation_importance_computation(self):
        """Test permutation-based importance computation."""
        importance_scores = self.analyzer._compute_permutation_importance(
            model=self.mock_model,
            inputs=self.sample_input,
            labels=self.sample_label
        )
        
        self.assertIsInstance(importance_scores, list)
        self.assertEqual(len(importance_scores), self.sample_input.shape[1])
        
        for score in importance_scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
    
    def test_integrated_gradients_computation(self):
        """Test integrated gradients importance computation."""
        importance_scores = self.analyzer._compute_integrated_gradients(
            model=self.mock_model,
            inputs=self.sample_input,
            labels=self.sample_label,
            steps=5  # Reduced for testing
        )
        
        self.assertIsInstance(importance_scores, list)
        self.assertEqual(len(importance_scores), self.sample_input.shape[1])
        
        for score in importance_scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
    
    def test_token_extraction(self):
        """Test token extraction from input tensor."""
        tokens = self.analyzer._extract_tokens(self.sample_input[0])
        
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), self.sample_input.shape[1])
        
        for token in tokens:
            self.assertIsInstance(token, str)
    
    def test_attention_weight_extraction(self):
        """Test attention weight extraction."""
        outputs = self.mock_model(self.sample_input, labels=self.sample_label)
        
        attention_weights = self.analyzer._extract_attention_weights(
            model=self.mock_model,
            inputs=self.sample_input,
            outputs=outputs
        )
        
        self.assertIsInstance(attention_weights, list)
        self.assertEqual(len(attention_weights), self.sample_input.shape[1])
        
        for weight in attention_weights:
            self.assertIsInstance(weight, float)
            self.assertGreaterEqual(weight, 0.0)
        
        # Check normalization (weights should sum to approximately 1)
        total_weight = sum(attention_weights)
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_structure_attention_analysis(self):
        """Test email structure attention analysis."""
        attention_weights = [0.1] * self.sample_input.shape[1]
        
        structure_attention = self.analyzer._analyze_structure_attention(
            inputs=self.sample_input[0],
            attention_weights=attention_weights
        )
        
        self.assertIsInstance(structure_attention, dict)
        
        expected_structures = ['subject', 'body', 'sender', 'recipient', 'other']
        for structure in expected_structures:
            self.assertIn(structure, structure_attention)
            self.assertIsInstance(structure_attention[structure], float)
            self.assertGreaterEqual(structure_attention[structure], 0.0)
    
    def test_prediction_stability_computation(self):
        """Test prediction stability computation."""
        # Test stable predictions
        stable_predictions = [1, 1, 1, 1, 1]
        stability = self.analyzer._compute_prediction_stability(stable_predictions)
        self.assertEqual(stability, 1.0)
        
        # Test unstable predictions
        unstable_predictions = [1, 2, 3, 4, 5]
        stability = self.analyzer._compute_prediction_stability(unstable_predictions)
        self.assertEqual(stability, 0.0)
        
        # Test partially stable predictions
        partial_predictions = [1, 1, 2, 2, 2]
        stability = self.analyzer._compute_prediction_stability(partial_predictions)
        self.assertGreater(stability, 0.0)
        self.assertLess(stability, 1.0)
    
    def test_attention_stability_computation(self):
        """Test attention stability computation."""
        # Test stable attention
        stable_attention = [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]
        stability = self.analyzer._compute_attention_stability(stable_attention)
        self.assertAlmostEqual(stability, 1.0, places=5)
        
        # Test unstable attention
        unstable_attention = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        stability = self.analyzer._compute_attention_stability(unstable_attention)
        self.assertLess(stability, 1.0)
    
    def test_reasoning_efficiency_computation(self):
        """Test reasoning efficiency computation."""
        # Test efficient reasoning (increasing confidence, decreasing entropy)
        confidences = [0.3, 0.6, 0.9]
        entropies = [2.0, 1.0, 0.5]
        
        efficiency = self.analyzer._compute_reasoning_efficiency(confidences, entropies)
        
        self.assertIsInstance(efficiency, float)
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)
    
    def test_reasoning_consistency_computation(self):
        """Test reasoning consistency computation."""
        predictions = [1, 1, 1, 1]
        confidences = [0.5, 0.6, 0.7, 0.8]
        
        consistency = self.analyzer._compute_reasoning_consistency(predictions, confidences)
        
        self.assertIsInstance(consistency, float)
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)
    
    def test_generate_interpretability_report(self):
        """Test comprehensive interpretability report generation."""
        sample_inputs = [self.sample_input, self.sample_input.clone()]
        sample_labels = [self.sample_label, self.sample_label.clone()]
        sample_texts = ["Test email 1", "Test email 2"]
        
        report = self.analyzer.generate_interpretability_report(
            model=self.mock_model,
            sample_inputs=sample_inputs,
            sample_labels=sample_labels,
            sample_texts=sample_texts
        )
        
        self.assertIsInstance(report, dict)
        self.assertIn('num_samples', report)
        self.assertIn('attention_analyses', report)
        self.assertIn('reasoning_analyses', report)
        self.assertIn('summary_statistics', report)
        
        self.assertEqual(report['num_samples'], 2)
        self.assertEqual(len(report['attention_analyses']), 2)
        self.assertEqual(len(report['reasoning_analyses']), 2)


class TestAdvancedFeaturesIntegration(unittest.TestCase):
    """Integration tests for advanced features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(TORCH_AVAILABLE and ADVANCED_FEATURES_AVAILABLE, 
                        "PyTorch and advanced features not available")
    def test_ensemble_evaluation_integration(self):
        """Test integration between ensemble training and evaluation."""
        # Create mock models
        models = [MockMacBookEmailTRM() for _ in range(3)]
        
        # Create mock dataloader
        mock_dataloader = create_mock_dataloader(num_samples=30)
        
        if mock_dataloader is None:
            self.skipTest("Mock dataloader creation failed")
        
        # Test ensemble evaluation
        from models.ensemble import ModelEnsemble
        
        ensemble = ModelEnsemble(
            models=models,
            voting_strategy="soft",
            device="cpu"
        )
        
        category_names = [
            "Newsletter", "Work", "Personal", "Spam", "Promotional",
            "Social", "Finance", "Travel", "Shopping", "Other"
        ]
        
        results = ensemble.evaluate_ensemble(mock_dataloader, category_names)
        
        self.assertIsInstance(results, dict)
        self.assertIn('ensemble_accuracy', results)
        self.assertIn('individual_accuracies', results)
        self.assertIn('improvement_over_best', results)
    
    @unittest.skipUnless(TORCH_AVAILABLE and ADVANCED_FEATURES_AVAILABLE, 
                        "PyTorch and advanced features not available")
    def test_evaluation_interpretability_integration(self):
        """Test integration between evaluation and interpretability."""
        # Create evaluator and analyzer
        evaluator = ComprehensiveEmailEvaluator(
            output_dir=os.path.join(self.temp_dir, "eval_output")
        )
        
        tokenizer = MockEmailTokenizer()
        analyzer = EmailInterpretabilityAnalyzer(
            tokenizer=tokenizer,
            output_dir=os.path.join(self.temp_dir, "interp_output")
        )
        
        # Create mock model and data
        model = MockMacBookEmailTRM()
        mock_dataloader = create_mock_dataloader(num_samples=20)
        
        if mock_dataloader is None:
            self.skipTest("Mock dataloader creation failed")
        
        # Run evaluation
        eval_result = evaluator.evaluate_model(
            model=model,
            dataloader=mock_dataloader,
            device="cpu"
        )
        
        # Run interpretability analysis on a sample
        sample_input = torch.randint(0, 1000, (1, 15))
        sample_label = torch.randint(0, 10, (1,))
        
        attention_viz = analyzer.analyze_sample_attention(
            model=model,
            inputs=sample_input,
            labels=sample_label
        )
        
        reasoning_analysis = analyzer.analyze_reasoning_cycles(
            model=model,
            inputs=sample_input,
            labels=sample_label
        )
        
        # Verify both analyses completed successfully
        self.assertIsInstance(eval_result, ComprehensiveEvaluationResult)
        self.assertIsInstance(attention_viz, AttentionVisualization)
        self.assertIsInstance(reasoning_analysis, ReasoningCycleAnalysis)
        
        # Check that results are consistent
        self.assertGreater(eval_result.total_samples, 0)
        self.assertGreater(attention_viz.confidence, 0.0)
        self.assertGreater(reasoning_analysis.num_cycles, 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)