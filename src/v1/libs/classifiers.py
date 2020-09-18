from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import svm
from rich.console import Console
import numpy
from . import utils
from . import generate_roc
from . import generate_conf_mat

console = Console()


def log(type, name, start_time):
    if type == 'start':
        console.print(
            f"[white]Starting [blue]{name} -> [white]({utils.get_time_diff(start_time)}s)")
    else:
        console.print(
            f"[yellow]Finishing [blue]{name} -> [white]({utils.get_time_diff(start_time)}s)")


def classify_generic(classificator, data):
    start_time = data["start_time"]

    x_test = data["x_test"]
    y_test = data["y_test"]
    x_train = data["x_train"]
    y_train = data["y_train"]

    experiment_hash = data["experiment_hash"]

    name = classificator.__class__.__name__

    console.print(
        f"\n[yellow]Classificator: [blue]{classificator}\n")

    log('start', 'fit', start_time)
    classificator.fit(x_train, y_train)
    log('end', 'fit', start_time)

    log('start', 'predict', start_time)
    predict = classificator.predict(x_test)
    log('end', 'predict', start_time)

    log('start', 'f1_score', start_time)
    f1_score = utils.round_float(sklearn_f1_score(
        y_test, predict, labels=numpy.unique(predict), average='weighted'))
    log('end', 'f1_score', start_time)

    log('start', 'accuracy', start_time)
    accuracy = utils.round_float(accuracy_score(y_test, predict))
    log('end', 'accuracy', start_time)

    log('start', 'conf_mat', start_time)
    conf_mat = confusion_matrix(y_test, predict)
    log('end', 'conf_mat', start_time)

    log('start', 'save_conf_mat', start_time)
    generate_conf_mat.save_conf_mat(
        experiment_hash, classificator, name, x_test, y_test)
    log('end', 'save_conf_mat', start_time)

    log('start', 'classification_report', start_time)
    creport = classification_report(y_test, predict)
    log('end', 'classification_report', start_time)

    time = utils.round_float(utils.get_time_diff(start_time))

    result = {
        'f1_score': f1_score,
        'accuracy': accuracy,
        'conf_mat': conf_mat,
        'creport': creport,
        'time': time
    }

    return result


def classify_svm(data):
    classificator = svm.LinearSVC(**data['parameters'])
    return classify_generic(classificator, data)


def classify_knn(data):
    classificator = KNeighborsClassifier(**data['parameters'])
    return classify_generic(classificator, data)


def classify_naive_bayes(data):
    classificator = GaussianNB(**data['parameters'])
    return classify_generic(classificator, data)


def classify_lda(data):
    classificator = LinearDiscriminantAnalysis(**data['parameters'])
    return classify_generic(classificator, data)


def classify_logistic_regression(data):
    classificator = LogisticRegression(**data['parameters'])
    return classify_generic(classificator, data)


def classify_perceptron(data):
    classificator = Perceptron(**data['parameters'])
    return classify_generic(classificator, data)


def classify_tree(data):
    classificator = DecisionTreeClassifier(**data['parameters'])
    return classify_generic(classificator, data)


def classify_mlp(data):
    # normalizar com z-score
    classificator = MLPClassifier(**data['parameters'])
    return classify_generic(classificator, data)
