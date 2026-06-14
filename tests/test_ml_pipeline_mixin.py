"""
Tests for MlPipelinesMixin — ML tables in ContextStore.
"""


class TestMlPipelines:
    def test_create_pipeline(self, context_store):
        # Need a campaign first (create_campaign returns the ID string)
        campaign_id = context_store.create_campaign(description="Test ML")
        pipeline = context_store.create_ml_pipeline(
            campaign_id=campaign_id,
            name="Test Pipeline",
            task="embryo_stage_classification",
        )
        assert pipeline["name"] == "Test Pipeline"
        assert pipeline["status"] == "planned"

    def test_get_pipeline(self, context_store):
        campaign_id = context_store.create_campaign(description="Test ML")
        created = context_store.create_ml_pipeline(
            campaign_id=campaign_id,
            name="P1",
        )
        fetched = context_store.get_ml_pipeline(created["id"])
        assert fetched is not None
        assert fetched["name"] == "P1"

    def test_list_pipelines(self, context_store):
        campaign_id = context_store.create_campaign(description="Test ML")
        context_store.create_ml_pipeline(campaign_id=campaign_id, name="P1")
        context_store.create_ml_pipeline(campaign_id=campaign_id, name="P2")

        all_pipelines = context_store.list_ml_pipelines()
        assert len(all_pipelines) == 2

        by_campaign = context_store.list_ml_pipelines(campaign_id=campaign_id)
        assert len(by_campaign) == 2

    def test_update_pipeline(self, context_store):
        campaign_id = context_store.create_campaign(description="Test ML")
        created = context_store.create_ml_pipeline(
            campaign_id=campaign_id,
            name="P1",
        )
        updated = context_store.update_ml_pipeline(
            created["id"],
            status="training",
            best_accuracy=0.85,
        )
        assert updated["status"] == "training"
        assert updated["best_accuracy"] == 0.85

    def test_pipeline_with_configs(self, context_store):
        campaign_id = context_store.create_campaign(description="Test ML")
        pipeline = context_store.create_ml_pipeline(
            campaign_id=campaign_id,
            name="P1",
            model_config={"architecture": "resnet18", "num_classes": 8},
            training_config={"batch_size": 32, "epochs": 50},
        )
        assert pipeline["model_config"]["architecture"] == "resnet18"
        assert pipeline["training_config"]["batch_size"] == 32

    def test_get_nonexistent_pipeline(self, context_store):
        assert context_store.get_ml_pipeline("nonexistent") is None


class TestTrainingRuns:
    def test_create_run(self, context_store):
        campaign_id = context_store.create_campaign(description="Test ML")
        pipeline = context_store.create_ml_pipeline(
            campaign_id=campaign_id,
            name="P1",
        )
        run = context_store.create_training_run(
            pipeline_id=pipeline["id"],
            model_config={"architecture": "resnet18"},
        )
        assert run["status"] == "planned"
        assert run["model_config"]["architecture"] == "resnet18"

    def test_update_run(self, context_store):
        campaign_id = context_store.create_campaign(description="Test ML")
        pipeline = context_store.create_ml_pipeline(
            campaign_id=campaign_id,
            name="P1",
        )
        run = context_store.create_training_run(pipeline_id=pipeline["id"])
        updated = context_store.update_training_run(
            run["id"],
            status="training",
            current_epoch=5,
            val_accuracy=0.82,
        )
        assert updated["status"] == "training"
        assert updated["current_epoch"] == 5
        assert updated["val_accuracy"] == 0.82

    def test_list_runs(self, context_store):
        campaign_id = context_store.create_campaign(description="Test ML")
        pipeline = context_store.create_ml_pipeline(
            campaign_id=campaign_id,
            name="P1",
        )
        context_store.create_training_run(pipeline_id=pipeline["id"])
        context_store.create_training_run(pipeline_id=pipeline["id"])

        runs = context_store.list_training_runs(pipeline["id"])
        assert len(runs) == 2


class TestDataAssessments:
    def test_save_and_get(self, context_store):
        assessment = context_store.save_data_assessment(
            total_sessions=3,
            total_embryos=120,
            total_volumes=1200,
            annotated_embryos=80,
            stage_distribution={"early": 30, "comma": 25, "pretzel": 15},
            coverage_gaps=["pretzel underrepresented"],
        )
        assert assessment["total_embryos"] == 120
        assert assessment["stage_distribution"]["early"] == 30

        fetched = context_store.get_data_assessment(assessment["id"])
        assert fetched is not None
        assert fetched["total_embryos"] == 120

    def test_assessment_with_pipeline(self, context_store):
        campaign_id = context_store.create_campaign(description="Test ML")
        pipeline = context_store.create_ml_pipeline(
            campaign_id=campaign_id,
            name="P1",
        )
        assessment = context_store.save_data_assessment(
            pipeline_id=pipeline["id"],
            total_embryos=50,
        )
        assert assessment["pipeline_id"] == pipeline["id"]
