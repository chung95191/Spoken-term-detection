import sys
model_name = ''
import bin.a2v_model_ed2 as model
import bin.processing_mfcc as processing
import random
import numpy as np
import cPickle
import time
import os
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] =sys.argv[2]
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
###########function###########
def eval_map(label_data,result_data,model_path):
    subprocess.call('/home_local/kenin815/bin/trec_eval.8.1/trec_eval %s %s >%s'%(label_data,result_data,model_path+'map'),shell=True)
    f=open(model_path+"/map")

    for i in f:
        if i.split()[0]=='map':
            return i.split()[2]
    f.close()
def result_gen(score,data_id_list,name):
    result=[]
    for i,j in zip(score,data_id_list):
        result.append((j[0],j[1],i))
    f=open(name,"w")
    for i in result:
        f.write(i[0]+' Q0 '+i[1]+' 525 '+str(i[2])+' STANDARD\n')
    f.close()

if __name__ =='__main__':
    #####data_dir#####
    zero=[0.0 for i in range(39)]
    dir_audio2feat = sys.argv[3]
    dir_query2feat = sys.argv[4]
    dir_trainlabel = sys.argv[5]
    dir_audio2feat_val = sys.argv[6]
    dir_query2feat_val = sys.argv[7]
    dir_vallabel = sys.argv[8]
    
    
    train_flag = int(sys.argv[9])
    #####load data#####
    '''if train_flag :
        train_label=[]
        f=open(dir_trainlabel)
        for i in f:
    	    train_label.append((i.split()[0],i.split()[2],i.split()[3]))
        train_num=len(train_label)'''
    val_label=[]
    f=open(dir_vallabel)    
    for i in f:
	    val_label.append((i.split()[0],i.split()[2],i.split()[3]))
    ####para####
    audio_maxframe=int(sys.argv[10])
    query_maxframe=int(sys.argv[11])
    batch_size=int(sys.argv[12])
    test_batch=int(sys.argv[13])
    epoch_num=int(sys.argv[14])
    vec_dim=int(sys.argv[15])
    lr=float(sys.argv[16])
    layer=int(sys.argv[17])
    saved_model_name=sys.argv[18]
    hop_num = int(sys.argv[19])
    beta=0.001
    bidirect=False
    test_num=len(val_label)
    model_path=sys.argv[1]
    data_name=model_path+'/m_result.txt'

    loss_history=[]
    if train_flag:
        f=open(model_path+'model_detail','w')
        f.write(str(model_name)+' bidirection:'+str(bidirect)+'\n')
        f.write(str(layer)+' layer LSTM\n')
        f.write('vec_dim: '+str(vec_dim)+' audio_maxframe: '+str(audio_maxframe)+' query_maxframe: '+str(query_maxframe)+'\n')
        f.write('l2_weight:'+str(beta)+'\n')
        f.write('lr: '+str(lr)+'\n')
        f.close()


    ###model_init####
    print '\nbuliding model....'
    t1=time.time()
    m = model.Model(audio_frame_nums=audio_maxframe,query_frame_nums=query_maxframe,rnn_size=vec_dim,layer=layer,bidirection=bidirect)
    m.lr=lr
    m.beta=beta
    m.build_model()
    t2=time.time()
    print 'layer:',layer,"rnn_size",vec_dim
    print 'done,cost time=',t2-t1
    print '\nloading feature....'
    if train_flag:
        query_map,doc_map,val_query_map,val_doc_map=processing.init(query_maxframe,audio_maxframe,dir_query2feat,dir_audio2feat,dir_query2feat_val,dir_audio2feat_val)

        query_list=query_map.keys()
        audio_list=doc_map.keys()
    else:
        val_query_map = processing.readfilefeature(dir_query2feat_val,query_maxframe)
        val_doc_map = processing.readfilefeature(dir_audio2feat_val,audio_maxframe)
    print 'done'

    def validation(label,batch_size,query_map,doc_map,data_name,dir_label):
        test_num = len(label)
        val_gen=processing.data_gen(label,batch_size,query_maxframe,audio_maxframe,query_map,doc_map)
        data_num=0
        score = []
        val_acc=0
        val_loss=0
        print '\nvalidataion'
        while data_num < test_num:
            data=val_gen.next()
            data_gen_num=data[1].shape[0]
            val_output=m.predict(data[0][0],data[0][1],data[0][2],data[0][3])
            score += (-1*val_output[0]).tolist()
            val_loss = (val_output[1]*data_gen_num + val_loss*data_num)  /(data_gen_num+data_num) 
            data_num += data_gen_num
            sys.stdout.write('loss:'+'{:<10}'.format(str(val_loss)) \
                            +' '+str(data_num)+'/'+str(test_num)+'\r')
            sys.stdout.flush()
        result_gen(score,label,data_name)
        map_value=eval_map(dir_label,data_name,model_path)
        print '\nval:'
        print 'map: ',map_value,' loss: ',val_loss
        return val_loss,map_value

    def a2v_train(data_list,max_len,feat_map,query):
        gen = processing.a2v_feat_gen(data_list,batch_size,max_len,feat_map)
        train_num = len(data_list)
        r_loss=0.
        data_num=0
        l2_loss=0
        ts=time.time()
        while data_num < train_num:
            data = gen.next()
            data_gen_num = data[1].shape[0]
            if query:
                batch_loss = m.train_query_a2v(data[0],data[1]) 
            else:
                batch_loss = m.train_audio_a2v(data[0],data[1]) 
            r_loss = ( r_loss*data_num + batch_loss*data_gen_num ) / (data_num+data_gen_num)    
            data_num += data_gen_num
            sys.stdout.write('r_loss: '+'{:<10}'.format(str(r_loss)) \
                            +' '+str(data_num)+'/'+str(train_num)+'\r')
            sys.stdout.flush()
        td=time.time()
        print '\n',
        print 'cost_time=',td-ts
        return r_loss
    f_record=open(model_path+'/loss_record','w')
    if train_flag:
        print "training"
        for i in range(epoch_num):
        ######train#######
            print '\nepoch:',i
            random.shuffle(query_list)
            loss = a2v_train(query_list,query_maxframe,query_map,1)
            loss += a2v_train(audio_list,audio_maxframe,doc_map,0)
            val_loss = validation(val_label,test_batch,val_query_map,val_doc_map,data_name,dir_vallabel)
            f_record.write("epoch: "+str(i)+\
                       ' loss: '+str(loss)+ \
                       ' val_loss: '+str(val_loss)+'\n')
            f_record.flush()
            #if float(val_loss[1])>=0.6:
            m.save_model(model_path+'/weight/model_'+str(i)+'_'+str(val_loss[1]))
    else:
        print "\ntesting"
        print "model_name:",saved_model_name
        m.restore(saved_model_name)
        val_loss = validation(val_label,test_batch,val_query_map,val_doc_map,data_name,dir_vallabel)
