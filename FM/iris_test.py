# python 3.6 
# 用iris对 fm进行测试

import os 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from .fm import FM_model

# ==============================
# 一、数据加载
# ==============================
def get_dt():
    iris = load_iris()
    irir_dt = pd.DataFrame(iris.data)
    irir_dt.columns=['a', 'b', 'c', 'd']
    irir_dt['label'] = iris.target
    label_map = dict(zip(irir_dt.label.unique(), range(irir_dt.label.nunique())))
    irir_dt['label']  = irir_dt.label.map(label_map)
    for i in ['a', 'b', 'c', 'd']:
        print(f'now {i}')
        try:
            irir_dt[i ] = pd.qcut(irir_dt[i], 10, labels=list(range(10)) ,duplicates='drop').astype('object')
        except:
            irir_dt[i ] = pd.qcut(irir_dt[i], 10, labels=list(range(9)) ,duplicates='drop').astype('object')

    iris_spares_dt = pd.get_dummies(irir_dt[['a', 'b', 'c', 'd']])
    return train_test_split(iris_spares_dt, irir_dt['label'], test_size=0.2)



# ==============================
# 二、数据预处理
# ==============================
def preocess(x, y):
    y = tf.cast(tf.constant(y.values), dtype=tf.int32)
    x = tf.cast(tf.constant(x.values), dtype=tf.int32)
    on_hot_y = tf.one_hot(y, depth=3)
    return x, on_hot_y

def get_db(x, y):
    x, y = preocess(x, y)
    x , y = tf.cast(x, dtype=tf.float32), tf.cast(y, dtype=tf.float32)
    db_ = tf.data.Dataset.from_tensor_slices((x, y))
    suffle_num = x.shape[0]
    db_ = db_.shuffle(suffle_num)
    db_ = db_.batch(10)
    return db_

train_db = get_db(tr_x, tr_y)
test_db = get_db(te_x,  te_y)

# ==============================
# 三、模型训练与评估
# ==============================
# 模型构建
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
if __name__ == '__main__':
    tr_x, te_x, tr_y, te_y = get_dt()
    train_db = get_db(tr_x, tr_y)
    test_db = get_db(te_x,  te_y)

    fm_model = FM_model(inp_dim=38, outp_dim=3, k=10, activation='softmax')
    model, log_dict = fm_model.train(train_db, lr=0.03, l1=0., l2= 0.02, epochs = 55, print_eval=10
            , test_flg=True, test_db=test_db, log_dict={'te_loss_lst':[], 'tr_loss_lst':[], 'acc_lst':[]} )

    svc = LinearSVC(loss='squared_epsilon_insensitive')
    rf = RandomForestClassifier(class_weight ='balanced')
    svc.fit(tr_x, tr_y)
    rf.fit(tr_x, tr_y)
    rf_p = rf.predict(te_x)
    svc_p = rf.predict(te_x)
    print('rd_acc:', sum(rf_p == te_y)/len(te_y))
    print('svc_acc:', sum(svc_p == te_y)/len(te_y))

"""
...
<EPOCH:41-step:11>: tr_loss: 0.77262, tr_acc:90.00%, te_loss: 0.71665, acc:100.00%
<EPOCH:42-step:11>: tr_loss: 0.72914, tr_acc:90.00%, te_loss: 0.71331, acc:100.00%
<EPOCH:43-step:11>: tr_loss: 0.70957, tr_acc:100.00%, te_loss: 0.71013, acc:100.00%
<EPOCH:44-step:11>: tr_loss: 0.72849, tr_acc:100.00%, te_loss: 0.70709, acc:100.00%
<EPOCH:45-step:11>: tr_loss: 0.72811, tr_acc:100.00%, te_loss: 0.70418, acc:100.00%
<EPOCH:46-step:11>: tr_loss: 0.67915, tr_acc:100.00%, te_loss: 0.70140, acc:100.00%
<EPOCH:47-step:11>: tr_loss: 0.76788, tr_acc:90.00%, te_loss: 0.69874, acc:100.00%
<EPOCH:48-step:11>: tr_loss: 0.76780, tr_acc:90.00%, te_loss: 0.69619, acc:100.00%
<EPOCH:49-step:11>: tr_loss: 0.74231, tr_acc:100.00%, te_loss: 0.69374, acc:100.00%
<EPOCH:50-step:11>: tr_loss: 0.65924, tr_acc:100.00%, te_loss: 0.69139, acc:100.00%
<EPOCH:51-step:11>: tr_loss: 0.66295, tr_acc:100.00%, te_loss: 0.68913, acc:100.00%
<EPOCH:52-step:11>: tr_loss: 0.78200, tr_acc:80.00%, te_loss: 0.68696, acc:100.00%
<EPOCH:53-step:11>: tr_loss: 0.74962, tr_acc:90.00%, te_loss: 0.68487, acc:100.00%
<EPOCH:54-step:11>: tr_loss: 0.65690, tr_acc:100.00%, te_loss: 0.68286, acc:100.00%
<EPOCH:55-step:11>: tr_loss: 0.72583, tr_acc:80.00%, te_loss: 0.68092, acc:100.00%
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_epsilon_insensitive',
     max_iter=1000, multi_class='ovr', penalty='l2', random_state=None,
     tol=0.0001, verbose=0)

RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=None, oob_score=False,
            random_state=None, verbose=0, warm_start=False)
rd_acc: 1.0
svc_acc: 1.0
"""
