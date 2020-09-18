import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import os


def save_conf_mat(folder, classifier, name, x_test, y_test):
    mk_folder = "result/conf_mat/" + folder
    if not os.path.exists(mk_folder):
        os.mkdir(mk_folder)
    output_file = mk_folder + "/" + name + ".png"
    disp = plot_confusion_matrix(classifier, x_test, y_test)
    disp.ax_.set_title("Confusion Matrix")
    plt.savefig(output_file)
