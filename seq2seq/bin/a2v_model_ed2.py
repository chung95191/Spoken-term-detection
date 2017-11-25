import os
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
class Model():
    def __init__(self,audio_frame_nums=10,query_frame_nums=5,rnn_size=100,layer=1,bidirection=False):
        tf.reset_default_graph()
        self.audio_frame_nums = audio_frame_nums
        self.query_frame_nums = query_frame_nums
        self.rnn_size = rnn_size
        self.bi =bidirection
        self.input_net = {}
        self.output_net = {}
        
        self.fw_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size)
        self.fw_cell_decoder = tf.nn.rnn_cell.LSTMCell(self.rnn_size)
        if bidirection:
            self.bw_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size)
            self.bw_cell_decoder = tf.nn.rnn_cell.LSTMCell(self.rnn_size)
        if layer >1:
            self.fw_cell = tf.nn.rnn_cell.MultiRNNCell([self.fw_cell]*layer,state_is_tuple=True)
            self.fw_cell_decoder = tf.nn.rnn_cell.MultiRNNCell([self.fw_cell_decoder]*layer,state_is_tuple=True)
            if self.bi:
                self.bw_cell = tf.nn.rnn_cell.MultiRNNCell([self.bw_cell]*layer,state_is_tuple=True)
                self.bw_cell_decoder = tf.nn.rnn_cell.MultiRNNCell([self.bw_cell_decoder]*layer,state_is_tuple=True)
        if self.bi:
            self.Wf = tf.Variable(tf.random_normal([self.rnn_size*4,2])) ###
        else:
            self.Wf = tf.Variable(tf.random_normal([self.rnn_size*2,2])) ###
        self.bf = tf.Variable(tf.random_normal([2]))
        self.output_net['loss'] = 0.
        self.output_net['reconstruct_loss'] = 0.
        self.sess = tf.Session()
        self.alpha = 1.0
        self.beta = 1.0
        self.lr = 1e-2
        self.input_size = 64
    def build_model(self):
        self.input_net['audio'] = tf.placeholder(tf.float32,[None,self.audio_frame_nums,39],name='audio_feat')
        self.input_net['query'] = tf.placeholder(tf.float32,[None,self.query_frame_nums,39],name='query_feat')
        self.input_net['audio_length'] = tf.placeholder(tf.int32,[None,self.audio_frame_nums],name='audio_mask')
        self.input_net['query_length'] = tf.placeholder(tf.int32,[None,self.query_frame_nums],name='query_mask')
        self.input_net['label'] = tf.placeholder(tf.float32,[None,2],name='label')
        self.input_net['alpha'] = tf.placeholder(tf.float32,[],name='alpha')
        self.input_net['beta'] = tf.placeholder(tf.float32,[],name='beta')
        self.input_net['gamma'] = tf.placeholder(tf.float32,[],name='gama')
        self.input_net['lr'] = tf.placeholder(tf.float32,[],name='lr')
        batch_size = tf.shape(self.input_net['audio'])[0]
        self.audio_length_reduce = tf.reduce_sum(self.input_net['audio_length'],axis=1)
        
        
        with tf.variable_scope('encoder'):
            if self.bi:
                audio_output, encoder_state_audio = tf.nn.bidirectional_dynamic_rnn(self.fw_cell,self.bw_cell,self.input_net['audio'],sequence_length=self.audio_length_reduce,dtype=tf.float32)
                audio_output = tf.concat_v2(audio_output,2)
            else:
                self.audio_output, self.encoder_state_audio = tf.nn.dynamic_rnn(self.fw_cell,self.input_net['audio'],sequence_length=self.audio_length_reduce,dtype=tf.float32)
        
        with tf.variable_scope('encoder',reuse=True):
            if self.bi:
                query_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(self.fw_cell,self.bw_cell,self.input_net['query'],sequence_length=tf.reduce_sum(self.input_net['query_length'],axis=1),dtype=tf.float32)
                query_output = tf.concat_v2(query_output,2)
            else:
                self.query_output, self.encoder_state = tf.nn.dynamic_rnn(self.fw_cell,self.input_net['query'],sequence_length=tf.reduce_sum(self.input_net['query_length'],axis=1),dtype=tf.float32)
        last_query = last_relevant(self.query_output,tf.reduce_sum(self.input_net['query_length'],axis=1))
        last_audio = last_relevant(self.audio_output,tf.reduce_sum(self.input_net['audio_length'],axis=1))
    
        
        
        self.var = tf.trainable_variables() 
        self.output_net['l2_loss'] = tf.add_n([ tf.nn.l2_loss(v) for v in self.var if 'bias' not in v.name ]) 
        
        
        
        ## autoencoder query
        self.fw_cell_decoder = tf.nn.rnn_cell.OutputProjectionWrapper(self.fw_cell_decoder,39)

        batch_size_query = tf.shape(self.input_net['query'])[0]
        decoder_query_inputs = self.query_frame_nums*[tf.zeros([batch_size_query,39])] #+ tf.unpack(self.input_net['query'],axis=1)
        
        decoder_query_origin = [tf.zeros([batch_size_query,39])] + tf.unpack(self.input_net['query'],axis=1)
        self.output_net['reconstruct_query_loss'] = 0
        with tf.variable_scope('rnn_decoder'):
            decoder_query_output, _ = tf.nn.seq2seq.rnn_decoder(decoder_query_inputs[:-1],self.encoder_state,self.fw_cell_decoder)
        self.do = decoder_query_output    
        for i in range(len(decoder_query_output)):
            self.output_net['reconstruct_query_loss'] += tf.sqrt(tf.reduce_mean(tf.square(tf.sub(decoder_query_output[i], decoder_query_origin[i+1]) ))) *tf.unpack(tf.cast(self.input_net['query_length'],tf.float32),axis=1)[i]
          
        self.output_net['reconstruct_query_loss'] /= tf.cast(tf.reduce_sum(self.input_net['query_length'],axis=1),tf.float32)
        self.output_net['reconstruct_query_loss'] = tf.reduce_mean(self.output_net['reconstruct_query_loss'])

        #self.output_net['reconstruct_query_loss'] += self.output_net['l2_loss']
        
        
        self.output_net['audio_vec'] = last_audio
        self.output_net['query_vec'] = last_query

        self.output_net['predict'] = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(last_audio,last_query)),axis=1))
        
        
        
        
        
        ###autoeencoder audio
        self.decoder_audio_inputs = self.audio_frame_nums*[tf.zeros([batch_size,39])] #+ tf.unpack(self.input_net['audio'],axis=1)
        self.decoder_audio_origin = [tf.zeros([batch_size,39])] + tf.unpack(self.input_net['audio'],axis=1)
        self.output_net['reconstruct_audio_loss'] = 0
        with tf.variable_scope('rnn_decoder',reuse=True):
            self.decoder_audio_output, _ = tf.nn.seq2seq.rnn_decoder(self.decoder_audio_inputs[:-1],self.encoder_state_audio,self.fw_cell_decoder)
        for i in range(len(self.decoder_audio_output)):
            self.output_net['reconstruct_audio_loss'] += tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.decoder_audio_output[i], self.decoder_audio_origin[i+1]) ))) *tf.unpack(tf.cast(self.input_net['audio_length'],tf.float32),axis=1)[i]
        self.output_net['reconstruct_audio_loss'] /= tf.cast(tf.reduce_sum(self.input_net['audio_length'],axis=1),tf.float32)
        self.output_net['reconstruct_audio_loss'] = tf.reduce_mean(self.output_net['reconstruct_audio_loss'])


        #self.output_net['reconstruct_audio_loss'] += self.output_net['l2_loss']
        print 'opti_query'
        self.joint_opti_query = tf.train.AdamOptimizer(self.lr)
        self.grads = self.joint_opti_query.compute_gradients(self.output_net['reconstruct_query_loss'])
        self.clip_grads = [(tf.clip_by_value(gv[0],-1,1), gv[1]) for gv in self.grads if gv[0] is not None]
        self.joint_opti_query = self.joint_opti_query.apply_gradients(self.clip_grads)  
        ###a2v_loss_audio
        print 'opti_audio'
        self.joint_opti_audio = tf.train.AdamOptimizer(self.lr)
        self.grads = self.joint_opti_audio.compute_gradients(self.output_net['reconstruct_audio_loss'])
        self.clip_grads = [(tf.clip_by_value(gv[0],-1,1), gv[1]) for gv in self.grads if gv[0] is not None]
        self.joint_opti_audio = self.joint_opti_audio.apply_gradients(self.clip_grads)  
        
        init = tf.global_variables_initializer()#tf.initialize_all_variables()
        print 'init'
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=1000)
    
    
    
    def train_audio_a2v(self,audio,audio_length):
        _ , loss = self.sess.run([self.joint_opti_audio,self.output_net['reconstruct_audio_loss']],\
                                  feed_dict={self.input_net['audio']:audio,\
                                             self.input_net['audio_length']:audio_length,\
                                             self.input_net['lr']:self.lr})
        return loss

    def train_query_a2v(self,query,query_length):
        _ , loss = self.sess.run([self.joint_opti_query,self.output_net['reconstruct_query_loss']],\
                                  feed_dict={self.input_net['query']:query,\
                                             self.input_net['query_length']:query_length,\
                                             self.input_net['lr']:self.lr})
        return loss

    def predict(self,audio,query,audio_length,query_length):
        return self.sess.run([self.output_net['predict'],self.output_net['reconstruct_audio_loss']],feed_dict={self.input_net['audio']:audio,self.input_net['query']:query,self.input_net['audio_length']:audio_length,self.input_net['query_length']:query_length})
    
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
    @lr.setter
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
