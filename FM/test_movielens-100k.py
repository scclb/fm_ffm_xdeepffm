# python 3.6
# create date: 2020-05-02


import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf 
from .fm import Mynetwork, FM_train

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
    
    user_features = df_user.columns.values.tolist()
    movie_features = df_item.columns.values.tolist()
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



if __name__ == '__main__':
    tr_x, tr_y, te_x, te_y = load_dataset()

    train_db = get_db(tr_x, tr_y)
    test_db = get_db(te_x, te_y)
    fm_model = Mynetwork(inp_dim=53, outp_dim=6, k=10, activation='softmax')
    model, log_dict = FM_train(fm_model, train_db, lr=0.01, l1=0.01, l2= 0.2, epochs = 10, print_eval=100
            , test_flg=True, test_db=test_db, log_dict={'te_loss_lst':[], 'tr_loss_lst':[], 'acc_lst':[]} )

    rf = RandomForestClassifier(class_weight ='balanced')
    tr_y_skl = np.argmax(tr_y, axis=1)
    te_y_skl = np.argmax(te_y, axis=1)
    rf.fit(tr_x.numpy(), tr_y_skl)
    rf_p = rf.predict(te_x.numpy())
    rf_ptr = rf.predict(tr_x.numpy())
    print('rf_acc:', sum(rf_p == te_y_skl)/len(te_y_skl))
    print('rf_tr_acc:', sum(rf_ptr == tr_y_skl)/len(tr_y_skl))

    lgb = lgb.LGBMClassifier()
    lgb.fit(tr_x.numpy(), tr_y_skl)
    lgb_p = rf.predict(te_x.numpy())
    print('lgb_acc:', sum(lgb_p == te_y_skl)/len(te_y_skl))

"""
<EPOCH:1-step:452>: tr_acc:30.59%,tr_loss: 1.68705, te_loss: 1.67483, acc:34.18%
<EPOCH:2-step:452>: tr_acc:27.65%,tr_loss: 1.70428, te_loss: 1.68250, acc:32.51%
<EPOCH:3-step:452>: tr_acc:35.88%,tr_loss: 1.64962, te_loss: 1.67272, acc:35.39%
<EPOCH:4-step:452>: tr_acc:30.00%,tr_loss: 1.68044, te_loss: 1.66995, acc:35.05%
<EPOCH:5-step:452>: tr_acc:31.18%,tr_loss: 1.68629, te_loss: 1.67070, acc:35.25%
<EPOCH:6-step:452>: tr_acc:40.59%,tr_loss: 1.63077, te_loss: 1.67008, acc:35.22%
<EPOCH:7-step:452>: tr_acc:36.47%,tr_loss: 1.64016, te_loss: 1.67100, acc:34.77%
<EPOCH:8-step:452>: tr_acc:33.53%,tr_loss: 1.66239, te_loss: 1.66680, acc:35.45%
<EPOCH:9-step:452>: tr_acc:31.18%,tr_loss: 1.68418, te_loss: 1.66772, acc:35.39%
<EPOCH:10-step:452>: tr_acc:38.82%,tr_loss: 1.64625, te_loss: 1.66757, acc:35.48%
rf_acc: 0.3337221633085896
rf_tr_acc: 0.9692944683670089
lgb_acc: 0.3337221633085896
"""
