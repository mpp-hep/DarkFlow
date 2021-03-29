import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr, fpr_PF, tpr_PF, fpr_H, tpr_H, fpr_O, tpr_O, fpr_T, tpr_T, fpr_I, tpr_I):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot(fpr_PF, tpr_PF, color='yellow', label='ROC_VAE+PF')
    plt.plot(fpr_H, tpr_H, color='red', label='ROC_VAE+HouseholderSNF')
    plt.plot(fpr_O, tpr_O, color='green', label='ROC_VAE+OrthoSNF')
    plt.plot(fpr_T, tpr_T, color='blue', label='ROC_VAE+TriSNF')
    plt.plot(fpr_I, tpr_I, color='brown', label='ROC_VAE+IAF')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

kl = np.load('kl.npy', allow_pickle=True)
loss = np.load('loss.npy', allow_pickle=True)
rec = np.load('rec.npy', allow_pickle=True)

kl_PF = np.load('PNF_kl_2EP.npy', allow_pickle=True)
loss_PF = np.load('PNF_loss_2EP.npy', allow_pickle=True)
rec_PF = np.load('PNF_rec_2EP.npy', allow_pickle=True)

kl_H = np.load('House_kl_2EP_BEST.npy', allow_pickle=True)
loss_H = np.load('House_loss_2EP_BEST.npy', allow_pickle=True)
rec_H = np.load('House_rec_2EP_BEST.npy', allow_pickle=True)

kl_O = np.load('Ortho_kl_2EP_BEST.npy', allow_pickle=True)
loss_O = np.load('Ortho_loss_2EP_BEST.npy', allow_pickle=True)
rec_O = np.load('Ortho_rec_2EP_BEST.npy', allow_pickle=True)

kl_T = np.load('Tri_kl_2EP_BEST.npy', allow_pickle=True)
loss_T = np.load('Tri_loss_2EP_BEST.npy', allow_pickle=True)
rec_T = np.load('Tri_rec_2EP_BEST.npy', allow_pickle=True)

kl_I = np.load('IAF_kl_2EP_BEST.npy', allow_pickle=True)
loss_I = np.load('IAF_loss_2EP_BEST.npy', allow_pickle=True)
rec_I = np.load('IAF_rec_2EP_BEST.npy', allow_pickle=True)


# kl_sm = kl[:2013]
# kl_bsm = kl[2013:]
# loss_sm = loss[:2013]
# loss_bsm = loss[2013:]
# rec_sm = rec[:2013]
# rec_bsm = rec[2013:]

# PF_kl_sm = kl_PF[:2013]
# PF_kl_bsm = kl_PF[2013:]
PF_loss_sm = loss_I[:2013]
PF_loss_bsm = loss_I[2013:]
# PF_rec_sm = rec_PF[:2013]
# PF_rec_bsm = rec_PF[2013:]

y_true = np.concatenate((np.zeros(2013),np.ones(2012)))

loss_nm = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
loss_PF_nm = (loss_PF - np.min(loss_PF)) / (np.max(loss_PF) - np.min(loss_PF))
loss_H_nm = (loss_H - np.min(loss_H)) / (np.max(loss_H) - np.min(loss_H))
loss_O_nm = (loss_O - np.min(loss_O)) / (np.max(loss_O) - np.min(loss_O))
loss_T_nm = (loss_T - np.min(loss_T)) / (np.max(loss_T) - np.min(loss_T))
loss_I_nm = (loss_I - np.min(loss_I)) / (np.max(loss_I) - np.min(loss_I))
x_nm = np.arange(1,4026)


auc = roc_auc_score(y_true, loss_nm)
auc_PF = roc_auc_score(y_true, loss_PF_nm)
auc_H = roc_auc_score(y_true, loss_H_nm)
auc_O = roc_auc_score(y_true, loss_O_nm)
auc_T = roc_auc_score(y_true, loss_T_nm)
auc_I = roc_auc_score(y_true, loss_I_nm)

fpr, tpr, thresholds = roc_curve(y_true, loss_nm)
fpr_PF, tpr_PF, thresholds_PF = roc_curve(y_true, loss_PF_nm)
fpr_H, tpr_H, thresholds_H = roc_curve(y_true, loss_H_nm)
fpr_O, tpr_O, thresholds_O = roc_curve(y_true, loss_O_nm)
fpr_T, tpr_T, thresholds_T = roc_curve(y_true, loss_T_nm)
fpr_I, tpr_I, thresholds_I = roc_curve(y_true, loss_I_nm)

print('AUC VAE: %.2f' % auc)#, '; Optimal Threshold:', thresholds[np.argmax(tpr - fpr)])
print('AUC VAE + PF : %.2f' % auc_PF)#, '; Optimal Threshold:', thresholds_PF[np.argmax(tpr_PF - fpr_PF)])
print('AUC VAE + HouseholderSNF : %.2f' % auc_H)#, '; Optimal Threshold:', thresholds_H[np.argmax(tpr_H - fpr_H)])
print('AUC VAE + OrthoSNF : %.2f' % auc_O)#, '; Optimal Threshold:', thresholds_O[np.argmax(tpr_O - fpr_O)])
print('AUC VAE + TriSNF : %.2f' % auc_T)#, '; Optimal Threshold:', thresholds_T[np.argmax(tpr_T - fpr_T)])
print('AUC VAE + IAF : %.2f' % auc_I)#, '; Optimal Threshold:', thresholds_I[np.argmax(tpr_I - fpr_I)])

plot_roc_curve(fpr, tpr, fpr_PF, tpr_PF, fpr_H, tpr_H, fpr_O, tpr_O, fpr_T, tpr_T, fpr_I, tpr_I)

x = np.arange(1,2014)
x1 = np.arange(1,2013)
plt.xlabel('Event')
plt.ylabel('PF_Loss after 2 epochs')
# plt.plot(x,kl_sm,label='kl SM Event')
# plt.plot(x1,kl_bsm,label='kl BSM Event')
plt.plot(x,PF_loss_sm,label='loss SM Event')
plt.plot(x1,PF_loss_bsm,label='loss BSM Event')
# plt.plot(x,rec_sm,label='rec SM Event')
# plt.plot(x1,rec_bsm,label='rec BSM Event')
plt.legend()
plt.show()

# plt.xlabel('Event')
# plt.ylabel('Loss after 2 epochs')
# plt.plot(x_nm,loss_nm,label='kl nm')
# plt.plot(x_nm,score,label='score')
# # plt.plot(x1,kl_bsm,label='kl BSM Event')
# plt.plot(x,loss_sm,label='loss SM Event')
# plt.plot(x1,loss_bsm,label='loss BSM Event')
# # plt.plot(x,rec_sm,label='rec SM Event')
# # plt.plot(x1,rec_bsm,label='rec BSM Event')
# plt.legend()
plt.show()