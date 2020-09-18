import lib.orchestrator as orchestrator
import lib.tabulation as tabulation
import lib.svmlight_utils as svmlight_utils
import lib.classifiers as classifiers
import lib.utils as utils
import time

from sklearn.datasets import load_svmlight_file
from rich.console import Console
from rich.table import Table

FILE_ORCHESTRATOR = 'orchestrator/index.json'

experiment_hash = utils.generate_hash()

console = Console()

output_table = {}


def print_result(classifier, result):
    execution_time = result["time"]
    table_results = Table(show_header=True, header_style="bold magenta")
    table_results.add_column("Type")
    table_results.add_column("Score")
    table_results.add_row("F1Score", str(result["f1_score"]))
    table_results.add_row("Accuracy", str(result["accuracy"]))
    table_results.add_row("Precision", str(result["precision"]))
    table_results.add_row("Recall", str(result["recall"]))
    console.print("\n")
    console.print(result["creport"])
    console.print("\n")
    console.print(table_results)
    console.print("\n")
    console.print(result["conf_mat"])
    console.print("\n")
    console.print(
        f"[bold red]Execution Time(s): ([white]{execution_time}[red])")


def save_result(configs, classifier, result):
    tabulation.save_tabulation_conf_mat(
        configs["result_conf_mat"], classifier, result["conf_mat"], experiment_hash)
    output_table["tabulation_writer"].writerow(
        [classifier, result["f1_score"], result["accuracy"], result["precision"], result["recall"], result["time"]])


def run_orchestrator(configs, experiments):
    start_time = time.time()
    console.print(
        f"[bold red]Starting Classifier ([white]Hash: {experiment_hash}[red])\n\n")
    console.print("[yellow]Starting Loading Files\n")
    x_train, y_train = load_svmlight_file(configs["svmlight_train_input"])
    x_test, y_test = load_svmlight_file(configs["svmlight_test_input"])
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    console.print(
        f"[yellow]Finishing Loading Input Files [white bold]({utils.get_time_diff(start_time)}s)\n")
    console.print(
        f"[green]Starting Experiment\n")
    for experiment in experiments:
        classifier = experiment['classifier']
        classifier_method = 'classify_' + classifier
        method_to_call = getattr(classifiers, classifier_method)
        data = {
            'parameters': experiment['parameters'],
            'experiment_hash': experiment_hash,
            'start_time': start_time,
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
        }
        result = method_to_call(data)
        start_time = time.time()
        print_result(classifier, result)
        save_result(configs, classifier, result)
    console.print(
        f"\n[green]Finishing Experiments")


if __name__ == "__main__":
    orchestrator = orchestrator.get_orchestrator(FILE_ORCHESTRATOR)
    configs = orchestrator["configs"]
    output_table = tabulation.get_output_table(configs, experiment_hash)
    experiments = orchestrator["experiments"]
    run_orchestrator(configs, experiments)
    output_table["tabulation_file"].close()
