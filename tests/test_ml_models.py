"""
Tests for ML data models — MLPipeline, TrainingRun, ModelConfig, etc.
"""

from gently.ml.models import (
    DataSplit,
    MLPipeline,
    ModelArchitectureType,
    ModelConfig,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)


class TestModelConfig:
    def test_round_trip(self):
        mc = ModelConfig(
            architecture="efficientnet_b2",
            num_classes=8,
            pretrained=True,
            input_channels=1,
        )
        d = mc.to_dict()
        mc2 = ModelConfig.from_dict(d)
        assert mc2.architecture == "efficientnet_b2"
        assert mc2.num_classes == 8
        assert mc2.pretrained is True
        assert mc2.input_channels == 1

    def test_defaults(self):
        mc = ModelConfig.from_dict({})
        assert mc.architecture == "resnet18"
        assert mc.num_classes == 8
        assert mc.dropout == 0.2


class TestTrainingConfig:
    def test_round_trip(self):
        tc = TrainingConfig(batch_size=64, epochs=100, learning_rate=3e-4, mixed_precision=True)
        d = tc.to_dict()
        tc2 = TrainingConfig.from_dict(d)
        assert tc2.batch_size == 64
        assert tc2.epochs == 100
        assert tc2.learning_rate == 3e-4
        assert tc2.mixed_precision is True

    def test_augmentations(self):
        tc = TrainingConfig()
        assert "random_horizontal_flip" in tc.augmentations


class TestDataSplit:
    def test_round_trip(self):
        ds = DataSplit(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=123)
        d = ds.to_dict()
        ds2 = DataSplit.from_dict(d)
        assert ds2.train_ratio == 0.8
        assert ds2.random_seed == 123

    def test_ratios_sum_to_one(self):
        ds = DataSplit()
        assert abs(ds.train_ratio + ds.val_ratio + ds.test_ratio - 1.0) < 0.01


class TestTrainingRun:
    def test_round_trip(self):
        run = TrainingRun(
            id="run1",
            pipeline_id="pipe1",
            status=TrainingStatus.TRAINING.value,
            current_epoch=10,
            total_epochs=50,
            val_accuracy=0.85,
            best_val_accuracy=0.87,
            model_config=ModelConfig(architecture="resnet18"),
        )
        d = run.to_dict()
        run2 = TrainingRun.from_dict(d)
        assert run2.id == "run1"
        assert run2.status == "training"
        assert run2.current_epoch == 10
        assert run2.model_config.architecture == "resnet18"

    def test_none_configs(self):
        run = TrainingRun.from_dict({"id": "r1"})
        assert run.model_config is None
        assert run.training_config is None


class TestMLPipeline:
    def test_round_trip(self):
        pipeline = MLPipeline(
            id="p1",
            campaign_id="c1",
            name="Embryo Classifier",
            task="embryo_stage_classification",
            model_config=ModelConfig(architecture="efficientnet_b2"),
            best_accuracy=0.94,
        )
        d = pipeline.to_dict()
        p2 = MLPipeline.from_dict(d)
        assert p2.id == "p1"
        assert p2.name == "Embryo Classifier"
        assert p2.model_config.architecture == "efficientnet_b2"
        assert p2.best_accuracy == 0.94


class TestTrainingStatus:
    def test_enum_values(self):
        assert TrainingStatus.PLANNED.value == "planned"
        assert TrainingStatus.TRAINING.value == "training"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.FAILED.value == "failed"


class TestModelArchitectureType:
    def test_enum_values(self):
        assert ModelArchitectureType.RESNET_18.value == "resnet18"
        assert ModelArchitectureType.EFFICIENTNET_B2.value == "efficientnet_b2"
