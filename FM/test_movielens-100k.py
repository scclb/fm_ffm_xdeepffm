# python 3.6
# create date: 2020-05-02
# 数据参考： https://github.com/LLSean/data-mining/tree/master/fm/data

import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf 
from .fm import FM_model

def load_dataset():
    data_path = r'data\fm'
    os.chdir(data_path)
    header = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    df_user = pd.read_csv('data/u.user', sep='|', names=header)
    header = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children',
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
            'Thriller', 'War', 'Western']
    df_item = pd.read_csv('data/u.item', sep='|', names=header, encoding = "ISO-8859-1")
    df_item = df_item.drop(columns=['title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown'])
    
    df_user['age'] = pd.cut(df_user['age'], [0,10,20,30,40,50,60,70,80,90,100], labels=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
    df_user = pd.get_dummies(df_user, columns=['gender', 'occupation', 'age'])
    df_user = df_user.drop(columns=['zip_code'])
    
    user_features = df_user.columns.tolist()
    movie_features = df_item.columns.tolist()
    cols = user_features + movie_features
    
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df_train = pd.read_csv('data/ua.base', sep='\t', names=header)
    df_train = df_train.merge(df_user, on='user_id', how='left') 
    df_train = df_train.merge(df_item, on='item_id', how='left')
    
    df_test = pd.read_csv('data/ua.test', sep='\t', names=header)
    df_test = df_test.merge(df_user, on='user_id', how='left') 
    df_test = df_test.merge(df_item, on='item_id', how='left')
    train_labels = tf.one_hot(df_train['rating'], depth=6)
    test_labels = tf.one_hot(df_test['rating'], depth=6)
    return tf.constant(df_train[cols].values), train_labels, tf.constant(df_test[cols].values), test_labels

def get_db(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    db_ = tf.data.Dataset.from_tensor_slices((x, y))
    suffle_num = x.shape[0]
    db_ = db_.shuffle(suffle_num)
    db_ = db_.batch(200)
    return db_


import lightgbm as lgb
import numpy as np
if __name__ == '__main__':
    tr_x, tr_y, te_x, te_y = load_dataset()

    train_db = get_db(tr_x, tr_y)
    test_db = get_db(te_x, te_y)
    fm_model = FM_model(inp_dim=53, outp_dim=6, k=10, activation='softmax')
    model, log_dict = fm_model.train(train_db, lr=0.01, l1=0.01, l2= 0.2, epochs = 10, print_eval=100
            , test_flg=True, test_db=test_db, log_dict={'te_loss_lst':[], 'tr_loss_lst':[], 'acc_lst':[]} )

    rf = RandomForestClassifier(class_weight ='balanced')
    tr_y_skl = np.argmax(tr_y, axis=1)
    te_y_skl = np.argmax(te_y, axis=1)
    rf.fit(tr_x.numpy(), tr_y_skl)
    rf_p = rf.predict(te_x.numpy())
    rf_ptr = rf.predict(tr_x.numpy())
    print('rf_acc:', sum(rf_p == te_y_skl)/len(te_y_skl))
    print('rf_tr_acc:', sum(rf_ptr == tr_y_skl)/len(tr_y_skl))

    svc = LinearSVC(loss='squared_epsilon_insensitive')
    svc.fit(tr_x.numpy(), tr_y_skl)
    svc_p = rf.predict(te_x.numpy())
    print('svc_acc:', sum(svc_p == te_y_skl)/len(te_y_skl))

    lgb_model = lgb.LGBMClassifier(reg_lambda=0.02, reg_alpha=0.01)
    lgb_model.fit(tr_x.numpy(), tr_y_skl)
    lgb_p = rf.predict(te_x.numpy())
    print('lgb_acc:', sum(lgb_p == te_y_skl)/len(te_y_skl))

"""
<EPOCH:1-step:452>: tr_loss: 1.73557, tr_acc:22.94%, te_loss: 1.72078, acc:25.96%
<EPOCH:2-step:452>: tr_loss: 1.67544, tr_acc:30.59%, te_loss: 1.67046, acc:34.66%
<EPOCH:3-step:452>: tr_loss: 1.68087, tr_acc:32.35%, te_loss: 1.67522, acc:34.27%
<EPOCH:4-step:452>: tr_loss: 1.69712, tr_acc:26.47%, te_loss: 1.69206, acc:29.73%
<EPOCH:5-step:452>: tr_loss: 1.63178, tr_acc:37.65%, te_loss: 1.66903, acc:34.92%
<EPOCH:6-step:452>: tr_loss: 1.70493, tr_acc:30.00%, te_loss: 1.67543, acc:33.97%
<EPOCH:7-step:452>: tr_loss: 1.64882, tr_acc:37.06%, te_loss: 1.66949, acc:34.89%
<EPOCH:8-step:452>: tr_loss: 1.66449, tr_acc:35.29%, te_loss: 1.67043, acc:35.09%
<EPOCH:9-step:452>: tr_loss: 1.65767, tr_acc:35.88%, te_loss: 1.66921, acc:34.99%
<EPOCH:10-step:452>: tr_loss: 1.65453, tr_acc:36.47%, te_loss: 1.66894, acc:35.09%
D:\Anacanda-python\install\lib\site-packages\sklearn\ensemble\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=None, oob_score=False,
            random_state=None, verbose=0, warm_start=False)
rf_acc: 0.3329798515376458
rf_tr_acc: 0.9692503036325494
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_epsilon_insensitive',
     max_iter=1000, multi_class='ovr', penalty='l2', random_state=None,
     tol=0.0001, verbose=0)
svc_acc: 0.3329798515376458
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        importance_type='split', learning_rate=0.1, max_depth=-1,
        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
        n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
        random_state=None, reg_alpha=0.01, reg_lambda=0.02, silent=True,
        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
lgb_acc: 0.3329798515376458
"""
