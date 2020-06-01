# python 3.6 tensorflow 2.0
# create date: 2020-06-01
# Function: fm
# tips:
"""
将 fm 前向反馈化简，然后将其认为是一个类全连接层

用Ftrl算法优化迭代更新
"""

from tensorflow.keras import layers

class FMdense(layers.Layer):
    def __init__(self, inp_dim, outp_dim, k, activation):
        super(FMdense, self).__init__()
        # 创建权值张量并添加
        self.activation = activation
        self.keneral = self.add_variable('w1', [inp_dim, outp_dim]
                                        ,trainable=True)
        self.v =  self.add_variable('v', [inp_dim, k]
                                        ,trainable=True)
        self.b = self.add_variable('b', [outp_dim]
                                        ,trainable=True)

    def call(self, x):
        y_out = (x@self.keneral  + self.b
        + 0.5 * tf.reduce_sum(
            (tf.pow( x@self.v, 2)  -  tf.pow(x,2) @ tf.pow(self.v, 2))
            ,axis=1, keepdims=True))
        return self.active(y_out)

    def active(self, y_out):
        if self.activation == 'sigmoid':
            return  tf.nn.sigmoid(y_out)
        if self.activation == 'softmax':
            return  tf.nn.softmax(y_out)


class Mynetwork(tf.keras.Model):
    def __init__(self, inp_dim, outp_dim, k=40, activation='softmax'):
        super(Mynetwork, self).__init__()
        self.fc1 = FMdense(inp_dim=inp_dim, outp_dim=outp_dim, k=k, activation=activation)
    
    def forward(self, x, training=None):
        x = self.fc1(x)
        return x 

    def corrects_(self, y_true, y_pred):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        return tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=tf.int32))

    def loss(self, y, out):
        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        return tf.reduce_mean(loss_func(y, out))

    def fm_optimizer(self, lr, l1, l2):
        return tf.keras.optimizers.Ftrl(learning_rate = lr
                        , l1_regularization_strength = l1
                        , l2_regularization_strength = l2)


def FM_train(model, dt_db, lr=0.01, l1=0.0, l2= 0.0, epochs = 50, print_eval=50
        , test_flg=False, test_db=None, log_dict={'te_loss_lst':[], 'tr_loss_lst':[], 'acc_lst':[]} ):
    optim_ = model.fm_optimizer(lr, l1, l2)
    for epoch in range(1, epochs+1):
        for step, (x, y) in enumerate(dt_db):
            with tf.GradientTape() as tape:
                out = model.forward(x)
                loss = model.loss(y, out)
            grads = tape.gradient(loss, model.trainable_variables)
            silence_ = optim_.apply_gradients(zip(grads, model.trainable_variables))

        
        if step % print_eval and step >= print_eval:
            if test_flg:
                corrects, total, loss_add = 0, 0, 0
                n = 0
                for xte, yte in test_db:
                    outi = model.forward(xte)
                    loss_add += model.loss(yte, outi).numpy()
                    corrects += model.corrects_(yte, outi)
                    total += xte.shape[0]
                    n += 1
                te_loss = loss_add/n
                acc = corrects/total
                log_dict['te_loss_lst'].append(te_loss)
                log_dict['tr_loss_lst'].append(loss.numpy())
                log_dict['acc_lst'].append(acc.numpy())
                print(f'<EPOCH:{epoch}-step:{step}>: tr_loss: {loss.numpy():.5f}, te_loss: {te_loss:.5f}, acc:{acc*100:.2f}%')
            else:
                acc = model.corrects(y, out) / x.shape[0]
                print(f'<EPOCH:{epoch}-step:{step}>: tr_loss: {loss.numpy():.5f}, acc:{acc*100:.2f}%')

    return  model, log_dict

