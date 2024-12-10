from typing import Optional
from src.core.config import DeploymentConfig

import mlflow

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentPipeline:
    """
    A class to handle the deployment pipeline for ML models.

    Attributes:
        _config (DeploymentConfig): Configuration settings for deployment.
        model_name (str): Name of the model to be deployed.
    """

    def __init__(self, config_file: DeploymentConfig):
        """
        Initializes the DeploymentPipeline with the given configuration file.

        Args:
            config_file (DeploymentConfig): The configuration file for deployment.
        """
        self._config = config_file
        self.model_name = config_file.model_name

    @staticmethod
    def get_all_experiments():
        """
        Retrieves all experiments in the MLflow tracking server.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing two lists - experiment names and experiment IDs.
        """
        experiment_names = []
        experiment_ids = []

        # Search for all experiments
        experiments = mlflow.search_experiments(view_type=mlflow.entities.ViewType.ALL)

        # Print experiment names and IDs
        for experiment in experiments:
            print(f"Name: {experiment.name}, ID: {experiment.experiment_id}")
            experiment_names.append(experiment.name)
            experiment_ids.append(experiment.experiment_id)
        return experiment_names, experiment_ids

    def _get_model_version_if_registered(
        self, run_id: str, client: mlflow.MlflowClient
    ):
        """
        Checks if the model is already registered and returns its version if it is.

        Args:
            run_id (str): The run ID of the model.
            client (mlflow.MlflowClient): The MLflow client.

        Returns:
            Optional[str]: The model version if it is registered, otherwise False.
        """
        registered_models = client.search_model_versions(f"name='{self.model_name}'")
        for model in registered_models:
            if run_id == model.run_id:
                return model.version
        return False

    def _register_model_and_set_tag(
        self,
        run_id: str,
        client: mlflow.MlflowClient,
    ):
        """
        Registers a new version of the model and sets an alias.

        Args:
            run_id (str): The run ID of the model.
            client (mlflow.MlflowClient): The MLflow client.
        """

        logger.info("Registering a new version")
        run_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(run_uri, self.model_name)
        version = result.version

        client.set_model_version_tag(
            self.model_name, version, "env", self._config.environment
        )

        logger.info(
            f"Model {self.model_name} version {version} is ready to be registered."
        )

    def register_model(
        self,
        experiment_id: int,
        run_id: Optional[str] = None,
        filter: Optional[str] = None,
        metric: Optional[str] = None,
        metric_mode: Optional[str] = "max",
    ):
        """
        Registers the model based on the specified mode and criteria.

        Args:
            experiment_id (int): The experiment ID.
            run_id (Optional[str]): The run ID. Required if mode is 'specific'.
            filter (Optional[str]): The filter string for searching runs.
            metric (Optional[str]): The metric to evaluate if mode is 'best'.
            metric_mode (Optional[str]): The mode of the metric evaluation ('max' or 'min'). Defaults to 'max'.

        Raises:
            ValueError: If 'metric' is not specified in 'best' mode or 'run_id' is not specified in 'specific' mode.
        """

        client = mlflow.MlflowClient()
        if self._config.mode == "last":
            try:
                run_id = mlflow.search_runs(
                    [experiment_id], filter_string=filter, order_by=["end_time DESC"]
                )["run_id"][0]
                logger.info(f"run_id {run_id} loaded.")
                self._register_model_and_set_tag(run_id, client)
                logger.info("Model was registered.")
            except KeyError as e:
                logger.error(
                    f"It wasnt possible to find experiments related to query {filter}"
                )

        elif self._config.mode == "best":
            if metric is None:
                raise ValueError("metric must be specified when mode is best")

            try:
                metric_order = "DESC" if metric_mode == "max" else "ASC"
                metric_filter = f"metrics.{metric} {metric_order}"
                run_id = mlflow.search_runs(
                    [experiment_id], filter_string=filter, order_by=[metric_filter]
                )["run_id"][0]
                self._register_model_and_set_tag(run_id, client)
            except KeyError as e:
                logger.error(
                    f"It wasnt possible to find experiments related to query {filter}"
                )

        elif self._config.mode == "specific":
            if run_id is None:
                raise ValueError("run_id must be specified when mode is specific")
            self._register_model_and_set_tag(run_id, client)

    @staticmethod
    def load_registered_model(model_name: str, version=None, latest=True):
        """
        Loads a registered model from MLflow or Azure ML.

        Args:
            model_name (str): The name of the registered model.
            version (Optional[str]): The version of the model to load. If None, the latest version is loaded.
            latest (bool): Whether to load the latest version of the model. Defaults to True.

        Returns:
            model: The loaded model.
        """
        client = mlflow.MlflowClient()

        if latest:
            model_metadata = client.get_latest_versions(model_name)
            version = model_metadata[0].version
        elif not latest and version is None:
            model_metadata = client.get_latest_versions(model_name)
            version = model_metadata[0].version

        model_uri = f"models:/{model_name}/{version}"
        mlflow_model = mlflow.pyfunc.load_model(model_uri)

        logger.info(f"Model path: {model_uri}")

        return mlflow_model, model_uri
