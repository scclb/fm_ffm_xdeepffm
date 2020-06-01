# python 3.6 
# 用iris对 fm进行测试

import os 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from .fm import Mynetwork, FM_train

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
if __name__ == '__main__':
    tr_x, te_x, tr_y, te_y = get_dt()
    train_db = get_db(tr_x, tr_y)
    test_db = get_db(te_x,  te_y)

    fm_model = Mynetwork(inp_dim=38, outp_dim=3, k=25, activation='softmax')
    model, log_dict = FM_train(fm_model, train_db, lr=0.01, l1=0., l2= 0.2, epochs = 20, print_eval=10
            , test_flg=True, test_db=test_db, log_dict={'te_loss_lst':[], 'tr_loss_lst':[], 'acc_lst':[]} )


    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(tr_x, tr_y)
    y_p = rf.predict(te_x)
    sum(y_p == te_y)/len(te_y)










