import multiprocessing

from AnalysisUtils.Evaluator import Evaluator
from AnalysisUtils.Experiment import Experiment

def experiment_processor(name: str, input_queue: multiprocessing.Queue,
                         output_queue: multiprocessing.Queue, evaluator: Evaluator):

    evaluator.init_databases(filename_prefix=name)

    while input_queue.qsize() > 0:
        full_path = input_queue.get()

        experiment = Experiment(full_path)
        evaluator.prepare_queries(experiment)
        if experiment.prompt_config["clusterings"] == "ground_truth":
            evaluator.evaluate_rules(experiment)
        evaluator.evaluate_mapping(experiment)

        output_queue.put((full_path, experiment))

