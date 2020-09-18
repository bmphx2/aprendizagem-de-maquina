from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os


def save_roc(folder, name, y_test, y_pred):
    mk_folder = "result/roc/" + folder
    if not os.path.exists(mk_folder):
        os.mkdir(mk_folder)
    output_file = mk_folder + "/" + name + ".png"
    output_general_file = mk_folder + "/" + "ROC.png"
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(name)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='roc ' + name)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve ' + name)
    plt.legend(loc='best')
    plt.savefig(output_file)

    plt.figure('general')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='roc ' + name)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve ' + name)
    plt.legend(loc='best')
    plt.savefig(output_general_file)
