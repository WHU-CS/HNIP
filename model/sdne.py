import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
# import rbm as rbm
from tqdm import tqdm

class SDNE:
    def __init__(self, config):
        self.is_varaibles_init = False
        self.config = config

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = tf_config)

        self.layers = len(config.struct)
        self.struct = config.struct
        self.sparse_dot = config.spare_dot
        self.W = {}
        self.b = {}
        struct = self.struct
        for i in range(self.layers - 1):
            name = 'encoder' + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name=name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name=name)

        struct.reverse()
        for i in range(self.layers - 1):
            name = 'decoder' + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name=name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name=name)
        self.struct.reverse()
        ############## define input ###################
        self.adjacent_matrix = tf.placeholder('float', [None, None])
        self.X = tf.placeholder('float', [None, config.struct[0]])

        self.__make_computer_graph()
        self.loss = self.__make_loss(config)
        self.optimizer = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.loss)

    def __make_computer_graph(self):
        def encoder(X):
            for i in range(self.layers - 1):
                name = 'encoder' + str(i)
                X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X

        def decoder(X):
            for i in range(self.layers - 1):
                name = 'decoder' + str(i)
                X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X

        self.H = encoder(self.X)
        self.X_reconstruct = decoder(self.H)

    def __make_loss(self, config):
        def get_2nd_loss(X, new_X, beta):
            B = X * (beta - 1) + 1
            N = tf.transpose(B)
            return tf.reduce_sum(tf.pow((new_X - X) * B, 2))


        def get_1st_loss(H, adj_mini_batch):

            D = tf.diag(tf.reduce_sum(adj_mini_batch, 1))
            L = D - adj_mini_batch
            return 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(H), L), H))

        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.values()])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.values()])
            return ret

        self.loss_2nd = get_2nd_loss(self.X, self.X_reconstruct, config.beta)
        self.loss_1st = get_1st_loss(self.H, self.adjacent_matrix)
        self.loss_reg = get_reg_loss(self.W, self.b)
        return config.gamma * self.loss_1st + config.alpha * self.loss_2nd + config.reg * self.loss_reg

    def save_model(self, path):
        saver = tf.train.Saver(self.b.values() + self.W.values())
        saver.save(self.sess, path)

    def restore_model(self, path):
        saver = tf.train.Saver(self.b.values() + self.W.values())
        saver.restore(self.sess, path)
        self.is_Init = True

    # apply the RBM for parameter initialization

    def do_variables_init(self, data):
        def assign(a,b):
            op = a.assign(b)
            self.sess.run(op)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        if self.config.restore_model:
            self.restore_model(self.config.restore_model)
            print('restore model' + self.config.restore_model)
        elif self.config.DBN_init:
            shape = self.struct
            myRBMs = []
            for i in range(len(shape) - 1):
                myRBM = rbm([shape[i], shape[i + 1]],{"batch_size": self.config.dbn_batch_size, "learning_rate": self.config.dbn_learning_rate})
                myRBMs.append(myRBM)
                for epoch in tqdm(range(self.config.dbn_epochs)):
                    error = 0
                    for batch in range(0, data.N, self.config.dbn_batch_size):
                        mini_batch = data.sample(self.config.dbn_batch_size).X
                        for k in range(len(myRBMs) - 1):
                            mini_batch = myRBMs[k].getH(mini_batch)
                        error += myRBM.fit(mini_batch)
                    # print("rbm epochs:", epoch, "error : ", error)

                W, bv, bh = myRBM.getWb()
                name = "encoder" + str(i)
                assign(self.W[name], W)
                assign(self.b[name], bh)
                name = "decoder" + str(self.layers - i - 2)
                assign(self.W[name], W.transpose())
                assign(self.b[name], bv)

        self.is_Init = True

    def __get_feed_dict(self, data):
        return {self.X: data.X, self.adjacent_matrix : data.adjacent_matrix}

    def fit(self, data):
        feed_dict = self.__get_feed_dict(data)
        ret, _ = self.sess.run((self.loss, self.optimizer), feed_dict = feed_dict)

    def get_loss(self, data):
        feed_dict = self.__get_feed_dict(data)
        return self.sess.run(self.loss, feed_dict=feed_dict)

    def get_embedding(self, data):
        return self.sess.run(self.H, feed_dict= self.__get_feed_dict(data))

    def get_W(self):
        return self.sess.run(self.W)

    def get_B(self):
        return self.sess.run(self.b)

    def close(self):
        self.sess.close()


class rbm:
    def __init__(self, shape, para):
        # shape[0] means the number of visible units
        # shape[1] means the number of hidden units
        self.para = para
        self.sess = tf.Session()
        stddev = 1.0 / np.sqrt(shape[0])
        self.W = tf.Variable(tf.random_normal([shape[0], shape[1]], stddev=stddev), name="Wii")
        self.bv = tf.Variable(tf.zeros(shape[0]), name="a")
        self.bh = tf.Variable(tf.zeros(shape[1]), name="b")
        self.v = tf.placeholder("float", [None, shape[0]])
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.buildModel()
        print("rbm init completely")
        pass

    def buildModel(self):
        self.h = self.sample(tf.sigmoid(tf.matmul(self.v, self.W) + self.bh))
        # gibbs_sample
        v_sample = self.sample(tf.sigmoid(tf.matmul(self.h, tf.transpose(self.W)) + self.bv))
        h_sample = self.sample(tf.sigmoid(tf.matmul(v_sample, self.W) + self.bh))
        lr = self.para["learning_rate"] / tf.to_float(self.para["batch_size"])
        W_adder = self.W.assign_add(
            lr * (tf.matmul(tf.transpose(self.v), self.h) - tf.matmul(tf.transpose(v_sample), h_sample)))
        bv_adder = self.bv.assign_add(lr * tf.reduce_mean(self.v - v_sample, 0))
        bh_adder = self.bh.assign_add(lr * tf.reduce_mean(self.h - h_sample, 0))
        self.upt = [W_adder, bv_adder, bh_adder]
        self.error = tf.reduce_sum(tf.pow(self.v - v_sample, 2))

    def fit(self, data):
        _, ret = self.sess.run((self.upt, self.error), feed_dict={self.v: data})
        return ret

    def sample(self, probs):
        return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

    def getWb(self):
        return self.sess.run([self.W, self.bv, self.bh])

    def getH(self, data):
        return self.sess.run(self.h, feed_dict={self.v: data})