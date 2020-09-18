import libs.utils as utils
import libs.tabulation as tabulation
import libs.classifiers as classifiers
import libs.orchestrator as orchestrator

import pandas as pd
import re
import time

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

from rich.console import Console
from rich.table import Table

FILE_ORCHESTRATOR = 'orchestrator/index.json'
FILE_TRAIN_INPUT_CSV = "data/train.csv"
FILE_TEST_INPUT_CSV = "data/test.csv"

start_time = 0
stemmer = PorterStemmer()
experiment_hash = utils.generate_hash()
console = Console()
output_table = {}
list_words = ['security', 'secure', 'vulnerable', 'leak', 'exception', 'crash', 'malicious',
              'sensitive', 'user', 'authentication', 'protect', 'vulnerability', 'authenticator', 'auth', 'npe']


def print_result(classifier, result):
    execution_time = result["time"]
    table_results = Table(show_header=True, header_style="bold magenta")
    table_results.add_column("Type")
    table_results.add_column("Score")
    table_results.add_row("F1Score", str(result["f1_score"]))
    table_results.add_row("Accuracy", str(result["accuracy"]))
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
        [classifier, result["f1_score"], result["accuracy"], result["time"]])


def prepare_dataset(dataset):
    data = []
    for i in range(dataset.shape[0]):
        commits = dataset.iloc[i, 1]
        commits = re.sub('[^A-Za-z]', ' ', commits)
        commits = commits.lower()
        tokenized_commits = word_tokenize(commits)
        commits_processed = []
        for word in tokenized_commits:
            if word not in set(stopwords.words('english')):
                commits_processed.append(stemmer.stem(word))
        commits_text = " ".join(commits_processed)
        data.append(commits_text)
    return data


def get_data(configs):
    train = pd.read_csv(configs["train"], encoding='utf-8')
    test = pd.read_csv(configs["test"], encoding='utf-8')
    data_train = prepare_dataset(train)
    data_test = prepare_dataset(test)
    tf_counter = TfidfVectorizer(max_features=100, stop_words='english',
                                 analyzer='word', use_idf=True, vocabulary=list_words)
    x_train = tf_counter.fit_transform(data_train).toarray()
    y_train = train.iloc[:, 0]
    x_test = tf_counter.fit_transform(data_test).toarray()
    y_test = test.iloc[:, 0]
    return (x_train, y_train, x_test, y_test)


def run_orchestrator(configs, experiments):
    start_time = time.time()
    console.print(
        f"[bold red]Starting Classifier ([white]Hash: {experiment_hash}[red])\n\n")
    console.print("[yellow]Starting Loading Files\n")
    x_train, y_train, x_test, y_test = get_data(configs)
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
