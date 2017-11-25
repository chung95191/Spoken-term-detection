import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import numpy as np
def norm(tensor):#normalzie last line
    return tensor/(tf.sqrt(tf.reduce_sum(tf.square(tensor),-1,keep_dims=True))+1e-12)
def cos(tensor1,tensor2):#by last dimension
    return tf.reduce_sum(tf.mul(norm(tensor1),norm(tensor2)),axis=-1)
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
def hopping(last_query,att_len,audio_output):

    att = tf.expand_dims(cos(audio_output,tf.expand_dims(last_query,1)),2)
    audio_mask=tf.expand_dims(tf.to_float(att_len),axis=2)                       
    att_norm = tf.mul(tf.exp(att),audio_mask)
    att_norm /=tf.reduce_sum(att_norm,1,keep_dims=True)   
    weight_sum = tf.reduce_sum(tf.mul(audio_output,att_norm),axis=1)
    return att_norm,weight_sum
class Model():
    def __init__(self,audio_frame_nums=10,query_frame_nums=5,rnn_size=100,layer=1,bidirection=False,hop=1):
        tf.reset_default_graph()
        self.audio_frame_nums = audio_frame_nums
        self.query_frame_nums = query_frame_nums
        self.rnn_size = rnn_size
        self.bi =bidirection
        self.input_net = {}
        self.output_net = {}
        self.hop = hop
        self.fw_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size)
        if bidirection:
            self.bw_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size)
        if layer >1:
            self.fw_cell = tf.nn.rnn_cell.MultiRNNCell([self.fw_cell]*layer,state_is_tuple=True)
            if self.bi:
                self.bw_cell = tf.nn.rnn_cell.MultiRNNCell([self.bw_cell]*layer,state_is_tuple=True)
        
        if self.bi:
            self.w1 = tf.Variable(tf.random_normal([self.rnn_size*4,128]))
            self.w2 = tf.Variable(tf.random_normal([128,64]))
            self.w3 = tf.Variable(tf.random_normal([64,32]))
            self.wf = tf.Variable(tf.random_normal([32,2])) ###
        else:
            self.w1 = tf.Variable(tf.random_normal([self.rnn_size*2,128]))
            self.w2 = tf.Variable(tf.random_normal([128,64]))
            self.w3 = tf.Variable(tf.random_normal([64,32]))
            self.wf = tf.Variable(tf.random_normal([32,1])) ###
        self.b1 = tf.Variable(tf.random_normal([128]))
        self.b2 = tf.Variable(tf.random_normal([64]))
        self.b3 = tf.Variable(tf.random_normal([32]))
        self.bf = tf.Variable(tf.random_normal([1]))
        self.output_net['loss'] = 0.
        self.output_net['reconstruct_loss'] = 0.
        self.sess = tf.Session()
        self.alpha = 1.0
        self.beta = 1.0
        self.lr = 1e-2
        self.input_size = 39
    def build_model(self):
        self.input_net['audio'] = tf.placeholder(tf.float32,[None,self.audio_frame_nums,39])
        self.input_net['query'] = tf.placeholder(tf.float32,[None,self.query_frame_nums,39])
        self.input_net['audio_length'] = tf.placeholder(tf.int32,[None,self.audio_frame_nums])
        self.input_net['query_length'] = tf.placeholder(tf.int32,[None,self.query_frame_nums])
        self.input_net['label'] = tf.placeholder(tf.float32,[None,1])
        self.input_net['alpha'] = tf.placeholder(tf.float32,[])
        self.input_net['beta'] = tf.placeholder(tf.float32,[])
        self.input_net['lr'] = tf.placeholder(tf.float32,[])
        batch_size = tf.shape(self.input_net['audio'])[0]
        self.audio_length_reduce = tf.reduce_sum(self.input_net['audio_length'],axis=1)
        
        with tf.variable_scope('encoder'):
            if self.bi:
                audio_output, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell,self.bw_cell,self.input_net['audio'],sequence_length=self.audio_length_reduce,dtype=tf.float32)
                audio_output = tf.concat_v2(audio_output,2)
            else:
                audio_output, _ = tf.nn.dynamic_rnn(self.fw_cell,self.input_net['audio'],sequence_length=self.audio_length_reduce,dtype=tf.float32)
        with tf.variable_scope('encoder',reuse=True):
            if self.bi:
                query_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(self.fw_cell,self.bw_cell,self.input_net['query'],sequence_length=tf.reduce_sum(self.input_net['query_length'],axis=1),dtype=tf.float32)
                query_output = tf.concat_v2(query_output,2)
            else:
                query_output, encoder_state = tf.nn.dynamic_rnn(self.fw_cell,self.input_net['query'],sequence_length=tf.reduce_sum(self.input_net['query_length'],axis=1),dtype=tf.float32)
        last_query = last_relevant(query_output,tf.reduce_sum(self.input_net['query_length'],axis=1))
        
        self.output_net['audio_vec'] = audio_output
        self.output_net['query_vec'] = last_query
        ##hopping
        
        self.att,hop_old = hopping(last_query,self.input_net['audio_length'],audio_output) 
        if self.hop==0:
            hop_new=hop_old
        else:
            hop_old += last_query
        for i in range(self.hop):
            self.att,hop_new = hopping(hop_old,self.input_net['audio_length'],audio_output)
            if i < self.hop-1:
                hop_new += hop_old
                hop_old = hop_new
        
        self.att1,hop1 = hopping(last_query,self.input_net['audio_length'],audio_output) 
        self.nn_input = tf.concat(1,[last_query,hop_new])
        self.nn_1 = tf.nn.elu(tf.add(tf.matmul(self.nn_input,self.w1),self.b1))
        self.nn_2 = tf.nn.elu(tf.add(tf.matmul(self.nn_1,self.w2),self.b2))
        self.nn_3 = tf.nn.elu(tf.add(tf.matmul(self.nn_2,self.w3),self.b3))
        self.output_net['predict'] = tf.add(tf.matmul(self.nn_3,self.wf),self.bf)
        self.output_net['dtw_loss'] = tf.sqrt(tf.square(tf.reduce_sum(tf.sub(self.output_net['predict'],self.input_net['label']),axis=1)))
        self.output_net['label_loss'] = tf.nn.softmax_cross_entropy_with_logits(self.output_net['predict'],self.input_net['label'])
        self.output_net['label_loss'] = tf.reduce_mean(self.output_net['label_loss'])
        
        self.var = tf.trainable_variables()
        self.lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.var if 'bias' not in v.name ]) *0.001
        self.output_net['loss'] = self.output_net['dtw_loss'] + self.lossL2
        self.joint_opti = tf.train.AdamOptimizer(self.lr)
        
        self.grads = self.joint_opti.compute_gradients(self.output_net['loss'])
        self.clip_grads = [(tf.clip_by_value(gv[0],-1,1), gv[1]) for gv in self.grads if gv[0] is not None]
        self.joint_opti = self.joint_opti.apply_gradients(self.clip_grads)  
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=1000)
    def train(self,audio,query,audio_length,query_length,label):
        _, loss = self.sess.run([self.joint_opti, \
                    (self.output_net['dtw_loss'])], \
                    feed_dict={self.input_net['audio']:audio \
                                ,self.input_net['query']:query \
                                ,self.input_net['audio_length']:audio_length \
                                ,self.input_net['query_length']:query_length \
                                ,self.input_net['label']:label \
                                ,self.input_net['alpha']:self.alpha \
                                ,self.input_net['beta']:self.beta \
                                ,self.input_net['lr']: self.lr})
        return loss
    def predict(self,audio,query,audio_length,query_length):
        return self.sess.run([self.output_net['predict']],feed_dict={self.input_net['audio']:audio,self.input_net['query']:query,self.input_net['audio_length']:audio_length,self.input_net['query_length']:query_length})
    def test(self,audio,query,audio_length,query_length,label):
        return self.sess.run([self.output_net['predict'],self.output_net['label_loss']],feed_dict={self.input_net['audio']:audio,self.input_net['query']:query,self.input_net['audio_length']:audio_length,self.input_net['query_length']:query_length,self.input_net['label']:label})
    def save_model(self,filename):
        self.saver.save(self.sess,filename)
    def restore(self,filename):
        self.saver.restore(self.sess,filename)#ckpt.model_checkpoint_path)
    def get_value(self,audio,query,audio_length,query_length):
        return self.sess.run([self.output_net['query_vec'],self.output_net['audio_vec']],feed_dict={self.input_net['audio']:audio,self.input_net['query']:query,self.input_net['audio_length']:audio_length,self.input_net['query_length']:query_length})
    @property
    def alpha(self):
        return self._alpha#,self.beta
    @alpha.setter
    def alpha(self,alpha):
        self.alpha = alpha
    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self,beta):
        self.beta = beta
    @property
    def lr(self):
        return self._lr
    @beta.setter
    def lr(self,lr):
        self.lr = lr
'''
test = Model()
test.build_model()
audio = np.random.rand(10,10,39)
query = np.random.rand(10,5,39)
audio_length = np.ones((10,10))
query_length = np.ones((10,5))
label = np.random.randint(2,size=(10,2))
test.train(audio,query,audio_length,query_length,label)
'''
