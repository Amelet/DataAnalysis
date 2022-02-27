from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss


def find_t(y_act, y_pred):
    """Function to find optimal threshold for predicted labels (y_pred),
    as returned by a classifier, to separate
    class 0/class 1"""
    k = np.arange(1, 11)
    f1 = []
    T_ks = []
    for kth in k:
        T_k = 0.1 * kth
        T_ks = np.append(T_ks, T_k)
        f1 = np.append(f1, f1_score(y_act, y_pred > T_k))
    prec, rec, thresh = precision_recall_curve(y_act, y_pred)
    return T_ks, f1, prec, rec, thresh


def weighted_log_loss(weight, actual, predicted):
    """Weighted log-loss
    False Negatives get higher score than False Positives"""
    wll = -1/len(actual)*np.sum(weight*actual*np.log(predicted)+(1-weight)*(1 - actual)*np.log(1-predicted))
    return wll


print("Difference in algorithms quality:")
algs = ["Ideal", "Typical", "Awful", "Avoids FP", "Avoids FN"]
data_files = ['y_ideal.csv', 'y_typical.csv', 'y_awful.csv', 'y_avoids_FP.csv', 'y_avoids_FN.csv']

for alg, file in zip(algs, data_files):
    print("\n",alg)
    y_pred = []
    y_act = []

    # 1. LOAD DATA
    with open(file, 'r') as f:
        lines = csv.reader(f, delimiter=',', quotechar='|')
        for line in lines:
            y_pred = np.append(y_pred, float(line[1]))
            y_act = np.append(y_act, float(line[0]))

    # 2. SET THRESHOLD
    T = 0.5  # threshold
    y_pred_t = y_pred>T
    weight = 0.3

    # 3. EXAMPLES OF METRICS: precision, recall, accuracy, f1
    print(f"""
    precision = {precision_score(y_act, y_pred_t)},
    recall = {recall_score(y_act, y_pred_t)}, 
    accuracy = {accuracy_score(y_act, y_pred_t)},
    f1 = {f1_score(y_act, y_pred_t)}, for T = {T},
    log-loss = {log_loss(y_act, y_pred)},
    weighted log-loss = {weighted_log_loss(weight, y_act, y_pred)}""")

    # 4. FIND a THRESHOLD T: VARY T, MEASURE F1, FIND T WHEN F1 MAX
    T_ks, f1, prec, rec, thresh = find_t(y_act, y_pred)

    # 5. PLOT RESULTS
    fig,ax = plt.subplots(1, 2)

    # 5.1 PRECISION AND RECALL
    ax[0].plot(thresh, prec[:-1], label="precision")
    ax[0].plot(thresh, rec[:-1], label="recall")
    ax[0].set_xlabel("threshold")
    ax[0].set_ylabel("precision/recall")
    ax[0].set_title(alg)
    ax[0].set_ylim(0, 1.1)
    ax[0].set_xlim(0, 1.1)
    ax[0].legend()

    # 5.2 F1 AS A FUNCTION OF THRESHOLD
    ax[1].plot(T_ks, f1)
    f1_max = np.where(f1 == np.max(f1))
    ax[1].scatter(T_ks[f1_max], f1[f1_max])
    ax[1].set_title(alg+"\noptimal threshold = "+str(np.round(T_ks[f1_max][0],2)))
    ax[1].set_xlabel("threshold")
    ax[1].set_ylabel("F1 metric")
    ax[1].set_ylim(0, 1.1)
    ax[1].set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()