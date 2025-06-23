import os
import pickle as pkl

from AnalysisUtils.Experiment import Experiment


class ExperimentCollection:

    collection_base_filename = "ExperimentCollection.pkl"

    def __init__(self, experiment_directory: str):

        self.filepath_to_experiment: dict[str, Experiment] = {}

        self._exp_collection_filepath = os.path.join(experiment_directory, self.collection_base_filename)
        if os.path.exists(self._exp_collection_filepath):
            with open(self._exp_collection_filepath, "rb") as f:
                self.filepath_to_experiment = pkl.load(f)

    def is_processed(self, full_path: str) -> bool:
        return full_path in self.filepath_to_experiment

    def add_experiment(self, full_path, experiment):
        self.filepath_to_experiment[full_path] = experiment

    def save(self):
        with open(self._exp_collection_filepath, "wb") as f:
            pkl.dump(self.filepath_to_experiment, f)