import sys
import numpy as np
import time
import theano
zero=[0.0 for i in range(39)]
def readfilefeature(file_dir,max_len):
    f=open(file_dir)
    feat_map={}
    for i in f:
        feat=[float(j) for j in i.split(":")[1].strip().split()]
        feat_len=len(feat)/39
        pad_num=max_len-feat_len
        
        if pad_num>=0:
            feat=feat+pad_num*zero
            
            one_array=np.ones(feat_len)
            zero_array=np.zeros(pad_num)
            feat_mask=np.concatenate((one_array,zero_array)).tolist()
        else:
            feat=feat[:39*max_len]
            feat_mask=np.ones(max_len).tolist()
            
        feat_map[i.split(':')[0].strip()]=(feat,feat_mask)
        
    f.close()
    return feat_map
def init(query_len,doc_len,dir_query2feat,dir_audio2feat,dir_query2feat_val,dir_audio2feat_val):
    ts=time.time()
    query_map=readfilefeature(dir_query2feat,query_len)
    doc_map=readfilefeature(dir_audio2feat,doc_len)
    val_query_map=readfilefeature(dir_query2feat_val,query_len)
    val_doc_map=readfilefeature(dir_audio2feat_val,doc_len)
    tend=time.time()
    print 'load_feature_time',tend-ts
    return query_map,doc_map,val_query_map,val_doc_map
    #return query_map,0,val_query_map,val_doc_map
def data_gen(data_list,batch_size,query_len,doc_len,query_map,doc_map):
    while 1:
        count=0
        query_f=[]
        query_len_f=[]
        doc_f=[]
        doc_len_f=[]
        label_f=[]
        for i in data_list:
            query=i[0]
            doc=i[1]
            label=i[2]
            ###query_feat###
            feat=query_map[query][0]
            feat_mask=query_map[query][1]
            
            
            query_f.append(feat)
            query_len_f.append(feat_mask)
            ####doc_feat
            feat=doc_map[doc][0]
            feat_mask=doc_map[doc][1]
            doc_f.append(feat)
            doc_len_f.append(feat_mask)
            ####label_feat####
            if i[2]=='1.0':
                label_f.append([1.0,0.0])
            else:
                label_f.append([0.0,1.0])
            count+=1
            if count==batch_size:
                query_f=np.array(query_f).reshape(batch_size,query_len,39)
                doc_f=np.array(doc_f).reshape(batch_size,doc_len,39)
                label_f=np.array(label_f)
                doc_len_f=np.array(doc_len_f)
                query_len_f=np.array(query_len_f)
                yield [doc_f,query_f,doc_len_f,query_len_f],label_f
                doc_f=[]
                query_f=[]
                doc_len_f=[]
                query_len_f=[]
                label_f=[]
                count=0
    
        residual_num=len(query_f)
        if residual_num>0:
            query_f=np.array(query_f).reshape(residual_num,query_len,39)
            doc_f=np.array(doc_f).reshape(residual_num,doc_len,39)
            label_f=np.array(label_f)
            doc_len_f=np.array(doc_len_f)
            query_len_f=np.array(query_len_f)
            yield [doc_f,query_f,doc_len_f,query_len_f],label_f
def gen_pair(data_pair_list,batch_size,query_len,doc_len,query_map,doc_map):

    while 1:
        count=0
        query_f=[]
        query_len_f=[]
        doc_f_pos=[]
        doc_len_f_pos=[]
        doc_f_neg=[]
        doc_len_f_neg=[]
        for i in data_pair_list:
            query = i[0]
            doc_pos = i[1]
            doc_neg = i[2]
            ###query_feat###
            feat=query_map[query][0]
            feat_mask=query_map[query][1]
            
            
            query_f.append(feat)
            query_len_f.append(feat_mask)
            ####doc_feat###
            feat=doc_map[doc_pos][0]
            feat_mask=doc_map[doc_pos][1]
            doc_f_pos.append(feat)
            doc_len_f_pos.append(feat_mask)

            feat=doc_map[doc_neg][0]
            feat_mask=doc_map[doc_neg][1]
            doc_f_neg.append(feat)
            doc_len_f_neg.append(feat_mask)

            count+=1
            if count==batch_size:
                query_f=np.array(query_f).reshape(batch_size,query_len,39)
                doc_f_pos=np.array(doc_f_pos).reshape(batch_size,doc_len,39)
                doc_f_neg=np.array(doc_f_neg).reshape(batch_size,doc_len,39)
                doc_len_f_pos=np.array(doc_len_f_pos)
                doc_len_f_neg=np.array(doc_len_f_neg)
                query_len_f=np.array(query_len_f)
                yield [doc_f_pos,doc_f_neg,query_f,doc_len_f_pos,doc_len_f_neg,query_len_f]
                doc_f_pos=[]
                doc_f_neg=[]
                query_f=[]
                doc_len_f_pos=[]
                doc_len_f_neg=[]
                query_len_f=[]
                count=0
        residual_num=len(query_f)
        if residual_num>0:
            query_f=np.array(query_f).reshape(residual_num,query_len,39)
            doc_f_pos=np.array(doc_f_pos).reshape(residual_num,doc_len,39)
            doc_len_f_pos=np.array(doc_len_f_pos)
            doc_f_neg=np.array(doc_f_neg).reshape(residual_num,doc_len,39)
            doc_len_f_neg=np.array(doc_len_f_neg)
            query_len_f=np.array(query_len_f)
            yield [doc_f_pos,doc_f_neg,query_f,doc_len_f_pos,doc_len_f_neg,query_len_f]
