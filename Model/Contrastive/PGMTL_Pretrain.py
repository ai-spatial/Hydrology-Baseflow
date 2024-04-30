%tensorflow_version 2.x
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:27:09 2022

@author: chens
"""

from __future__ import print_function, division
import numpy as np
import tensorflow.compat.v1 as tf
import os
import random
from sklearn.cluster import KMeans

tf.disable_v2_behavior() 

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'

tf.reset_default_graph()
random.seed(9001)


''' Declare constants '''
learning_rate = 0.002#0.001#0.005/3/2 default 0.002#0.0009
epochs = 60
# epochs_val =15
e_reduce = 15
input_size = 30
input_size_dy = input_size - 10
state_size = 30 
n_classes = 1
cv_idx = 2
N_seg = 60 
kb=1.0



npic = 10
n_steps = int(3600/npic)
n_steps1 = int(n_steps)
N_sec = (npic-1)*2+1


#setting up gage dataset dimemsion reduction
input_size_gage = 60
output_size_gage = 10

#apply grad
n_upd = 1
n_fupd = 1
upd_lr = 0.005

#recycle time:
recycle = 3

''' Build Graph '''
x_tr1 = tf.placeholder("float", [None, n_steps,input_size_dy]) 
gx_tr1 = tf.placeholder("float", [None, n_steps,input_size_gage]) 

x_tr2 = tf.placeholder("float", [None, n_steps,input_size_dy]) 
gx_tr2 = tf.placeholder("float", [None, n_steps,input_size_gage]) 

y_tr1 = tf.placeholder("float", [None, n_steps]) 
m_tr1 = tf.placeholder("float", [None, n_steps])

y_tr2 = tf.placeholder("float", [None, n_steps]) 
m_tr2 = tf.placeholder("float", [None, n_steps])

y_val1 = tf.placeholder("float", [None, n_steps,1]) 
y_val2 = tf.placeholder("float", [None, n_steps,1]) 
# y_val1_2 = tf.placeholder("float", [None, n_steps,1]) 
# y_val2_2 = tf.placeholder("float", [None, n_steps,1]) 



W1 = tf.get_variable('W_1',[input_size_gage, output_size_gage], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))#default 0.02
b1 = tf.get_variable('b_1',[output_size_gage],  tf.float32,
                                  initializer=tf.constant_initializer(0.0)) #default 0


# Wg2 = tf.get_variable('W_g2',[state_size, state_size], tf.float32,
#                                  tf.random_normal_initializer(stddev=0.02))
# bg2 = tf.get_variable('b_g2',[state_size],  tf.float32,
#                                  initializer=tf.constant_initializer(0.0))


Wi = tf.get_variable('Wi',[state_size, state_size], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))

Wf = tf.get_variable('Wf',[state_size, state_size], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))

Wo = tf.get_variable('Wo',[state_size, state_size], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))

Wg = tf.get_variable('Wg',[state_size, state_size], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))
# Ws = tf.get_variable('Ws',[state_size, state_size], tf.float32,
#                                  tf.random_normal_initializer(stddev=0.02))

Ui = tf.get_variable('Ui',[input_size, state_size], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))

Uf = tf.get_variable('Uf',[input_size, state_size], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))

Uo = tf.get_variable('Uo',[input_size, state_size], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))

Ug = tf.get_variable('Ug',[input_size, state_size], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))

# Us = tf.get_variable('Us',[input_size, state_size], tf.float32,
#                                  tf.random_normal_initializer(stddev=0.02))



w_fin = tf.get_variable('w_fin',[state_size, n_classes], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))
b_fin = tf.get_variable('b_fin',[n_classes],  tf.float32,
                                  initializer=tf.constant_initializer(0.0))

  


def forward(xi, gx, W1,b1,Wi,Wf,Wo,Wg,Ui,Uf,Uo,Ug,w_fin,b_fin,reuse=False):
    gx1 = tf.sigmoid(tf.matmul(gx,W1)+b1)
    x = tf.concat([xi, gx1], 2) 
    x = tf.reshape(x,[-1,n_steps1,state_size])
    o_sr = []
    
    i = tf.sigmoid(tf.matmul(x[:,0,:],Ui))
##      forget gate
#    f = tf.sigmoid(tf.matmul(x[:,0,:],Uf))
    #  output gate
    o = tf.sigmoid(tf.matmul(x[:,0,:],Uo))
    #  candidate cell
    g = tf.tanh(tf.matmul(x[:,0,:],Ug))
    
    c_pre = g*i
    h_pre = tf.tanh(c_pre)*o
    o_sr.append(h_pre)
            
    for t in range(1,n_steps1):
        i = tf.sigmoid(tf.matmul(x[:,t,:],Ui) + tf.matmul(h_pre,Wi))
        #  forget gate
        f = tf.sigmoid(tf.matmul(x[:,t,:],Uf) + tf.matmul(h_pre,Wf))
        #  output gate
        o = tf.sigmoid(tf.matmul(x[:,t,:],Uo) + tf.matmul(h_pre,Wo))
        
        #  candidate cell
        g = tf.tanh(tf.matmul(x[:,t,:],Ug) + tf.matmul(h_pre,Wg))
        

        c_pre = (c_pre)*f + g*i
        # output state
        h_pre = tf.tanh(c_pre)*o
        o_sr.append(h_pre)
        
    
    o_sr = tf.stack(o_sr,axis=1) # N_seg - T - state_size
    oh = tf.reshape(o_sr,[-1,state_size])
    
    
    pred = tf.matmul(oh,w_fin)+b_fin
    pred = tf.reshape(pred,[-1,n_steps,1])

    return o_sr, pred

def loss_measure(pred,y,m):
    pred_s = tf.reshape(pred,[-1,1])
    y_s = tf.reshape(y,[-1,1])
    m_s = tf.reshape(m,[-1,1])        
    r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s-y_s),m_s))+1e-20)/(tf.reduce_sum(m_s)+1))
    return r_cost


def apply_gr(W1,b1,Wi,Wf,Wo,Wg,Ui,Uf,Uo,Ug,w_fin,b_fin, grads):
    
    W1n = W1 - upd_lr*grads[0]
    b1n = b1 - upd_lr*grads[1]
    Win = Wi - upd_lr*grads[2]
    Wfn = Wf - upd_lr*grads[3]
    Won = Wo - upd_lr*grads[4]
    Wgn = Wg - upd_lr*grads[5]
    Uin = Ui - upd_lr*grads[6]
    Ufn = Uf - upd_lr*grads[7]
    Uon = Uo - upd_lr*grads[8]
    Ugn = Ug - upd_lr*grads[9]
    w_finn = w_fin - upd_lr*grads[10]
    b_finn = b_fin - upd_lr*grads[11]
 
    return W1n,b1n,Win,Wfn,Won,Wgn,Uin,Ufn,Uon,Ugn,w_finn,b_finn



h_tr,pred_tr = forward(x_tr1, gx_tr1, W1,b1,Wi,Wf,Wo,Wg,Ui,Uf,Uo,Ug,w_fin,b_fin)

cost_tr1 = loss_measure(pred_tr,y_tr1,m_tr1)



# define the contrastive loss


def loss_measure_ent(pred_tr, y_val1, y_val2):
  sim1 = tf.expand_dims(tf.reduce_sum(-tf.abs(pred_tr-y_val1),axis=-1),axis=2)
  sim2 = tf.expand_dims(tf.reduce_sum(-tf.abs(pred_tr-y_val2),axis=-1),axis=2)
    
  sim = tf.concat([sim1,sim2],axis=-1)
  sim_p = tf.nn.softmax(sim)
  sim = tf.reshape(sim,[-1,2])
  sim_l = tf.reshape(sim_p,[-1,2])
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sim_l,logits=sim))
  return cost

cost_ent = loss_measure_ent(pred_tr, y_val1, y_val2)
#total cost
cost = cost_tr1 + 0.1*cost_ent # 0.02, 0.5

tvars = tf.trainable_variables()
for i in tvars:
    print(i)


grads = tf.gradients(cost, tvars)
W1n,b1n,Win,Wfn,Won,Wgn,Uin,Ufn,Uon,Ugn,w_finn,b_finn = apply_gr(W1,b1,Wi,Wf,Wo,Wg,Ui,Uf,Uo,Ug,w_fin,b_fin, grads)

for it in range(1,n_upd):
    h1_tr,pred1_tr = forward(x_tr1, gx_tr1, W1n,b1n,Win,Wfn,Won,Wgn,Uin,Ufn,Uon,Ugn,w_finn,b_finn)
    cost_tr1 = loss_measure(pred1_tr,y_tr1,m_tr1)
    cost_ent = loss_measure_ent(h1_tr, y_val1, y_val2)
    cost = cost_tr1 + 0.1*cost_ent # 0.02, 0.5
    grads1 = tf.gradients(cost, [W1n,b1n,Win,Wfn,Won,Wgn,Uin,Ufn,Uon,Ugn,w_finn,b_finn]) 
    W1n,b1n,Win,Wfn,Won,Wgn,Uin,Ufn,Uon,Ugn,w_finn,b_finn = apply_gr(W1n,b1n,Win,Wfn,Won,Wgn,Uin,Ufn,Uon,Ugn,w_finn,b_finn,grads1)

#validating for adjust lambda
h2_tr,pred2_tr = forward(x_tr2, gx_tr2, W1n,b1n,Win,Wfn,Won,Wgn,Uin,Ufn,Uon,Ugn,w_finn,b_finn)
cost1 = loss_measure(pred2_tr,y_tr2,m_tr2) 


grads2 = tf.gradients(cost1, tvars)


optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(zip(grads2, tvars))

''' Load data '''


gage_feat = np.load('./data/gage_processing.npy')

feat = np.load('./data/NCA_processing.npy')
label = np.load('./data/station_log.npy')
mask = np.load('./data/synthetic_mask.npy')

mask[:,3600:7200] = 0 #for training

mask1 = np.load('./data/synthetic_mask.npy')
mask1[:,0:3600] = 0 #for validating 




labela = np.load('./data/simulated_eq1.npy')
labelb = np.load('./data/simulated_eq2.npy')



maska = np.load('./data/synthetic_mask.npy')
maskb = np.load('./data/synthetic_mask.npy')

maska1 = np.load('./data/synthetic_mask.npy')
maskb1 = np.load('./data/synthetic_mask.npy')


labela = np.log(labela)
labelb = np.log(labelb)

labela = labela*maska
labelb = labelb*maskb


labela = np.expand_dims(labela, axis=2)
labelb = np.expand_dims(labelb, axis=2)


#####doing clustering to otain different groups in first step
w = np.load('./step1/ws_pretrain.npy')

b = np.load('./step1/bs_pretrain.npy')

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
k = 10
init = gage_feat[:,0,:]
sample = sigmoid(np.matmul(init,w) + b)
c_index = np.random.randint(low=0, high=60, size=k)
np.save('./step2/cluster_index.npy', c_index)
center = sample[c_index,]

estimator = KMeans(n_clusters=k,init=center)
estimator.fit(sample)
cluster = estimator.labels_ 
centroids = estimator.cluster_centers_
center[:] = centroids[:]
np.save('./step2/center.npy', center)
np.save('./step2/cluster.npy', cluster)


pred_all = np.zeros((len(sample),3600))
for r in range(recycle):
    k_list = list(range(k))
    
    random.shuffle(k_list)
    epochs = epochs - e_reduce
    print('epoch iteration: ', epochs)
    
    for ct in k_list:
        print('cluster No.', ct)
        loc = np.where(cluster == ct)[0]
        N_seg = len(loc)
        
        x_te = feat[loc,cv_idx*3600:(cv_idx+1)*3600,:]
        gx_te = gage_feat[loc,cv_idx*3600:(cv_idx+1)*3600,:]
        y_te = label[loc,cv_idx*3600:(cv_idx+1)*3600]
        y_tea = labela[loc,cv_idx*3600:(cv_idx+1)*3600,:]
        y_teb = labelb[loc,cv_idx*3600:(cv_idx+1)*3600,:]
        m_te = mask[loc,cv_idx*3600:(cv_idx+1)*3600]
    
        
        
    
        if cv_idx==2:
            x_tr_1 = feat[loc,:3600,:]
            y_tr_1 = label[loc,:3600]
            y_tr_1a = labela[loc,:3600,:]
            y_tr_1b = labelb[loc,:3600,:]
            m_tr_1 = mask[loc,:3600]
            m_tr_1 = m_tr_1[loc,:3600]
            m1_tr_1 = mask1[loc,:3600]
            gx_tr_1 = gage_feat[loc,:3600,:]
            
            x_tr_2 = feat[loc,3600:2*3600,:]
            y_tr_2 = label[loc,3600:2*3600:]
            y_tr_2a = labela[loc,3600:2*3600:,:]
            y_tr_2b = labelb[loc,3600:2*3600:,:]
            m_tr_2 = mask[loc,3600:2*3600:]#for validating
            m_tr_2 = m_tr_2[loc,:3600]
            m1_tr_2 = mask1[loc,3600:2*3600:]
            gx_tr_2 = gage_feat[loc,3600:2*3600,:]
            
            
        x_train_1 = np.zeros([N_seg*N_sec,n_steps,input_size_dy])
        y_train_1 = np.zeros([N_seg*N_sec,n_steps])
        y_train_1a = np.zeros([N_seg*N_sec,n_steps,1])
        y_train_1b = np.zeros([N_seg*N_sec,n_steps,1])
        m_train_1 = np.zeros([N_seg*N_sec,n_steps])
        m1_train_1 = np.zeros([N_seg*N_sec,n_steps])
        gx_train_1 = np.zeros([N_seg*N_sec,n_steps,input_size_gage])
        
        x_train_2 = np.zeros([N_seg*N_sec,n_steps,input_size_dy])
        y_train_2 = np.zeros([N_seg*N_sec,n_steps])
        y_train_2a = np.zeros([N_seg*N_sec,n_steps,1])
        y_train_2b = np.zeros([N_seg*N_sec,n_steps,1])
        m_train_2 = np.zeros([N_seg*N_sec,n_steps])#for validating
        m1_train_2 = np.zeros([N_seg*N_sec,n_steps])
        gx_train_2 = np.zeros([N_seg*N_sec,n_steps,input_size_gage])
        
        x_test = np.zeros([N_seg*N_sec,n_steps,input_size_dy])
        y_test = np.zeros([N_seg*N_sec,n_steps])
        y_testa = np.zeros([N_seg*N_sec,n_steps,1])
        y_testb = np.zeros([N_seg*N_sec,n_steps,1])
        m_test = np.zeros([N_seg*N_sec,n_steps])
        gx_test = np.zeros([N_seg*N_sec,n_steps,input_size_gage])
        
        
        
        for i in range(1,N_sec+1):
            x_train_1[(i-1)*N_seg:i*N_seg,:,:]=x_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
            y_train_1[(i-1)*N_seg:i*N_seg,:]=y_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
            y_train_1a[(i-1)*N_seg:i*N_seg,:]=y_tr_1a[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
            y_train_1b[(i-1)*N_seg:i*N_seg,:]=y_tr_1b[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
            m_train_1[(i-1)*N_seg:i*N_seg,:]=m_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
            m1_train_1[(i-1)*N_seg:i*N_seg,:]=m1_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
            gx_train_1[(i-1)*N_seg:i*N_seg,:,:]=gx_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
            
            x_train_2[(i-1)*N_seg:i*N_seg,:,:]=x_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
            y_train_2[(i-1)*N_seg:i*N_seg,:]=y_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
            y_train_2a[(i-1)*N_seg:i*N_seg,:]=y_tr_2a[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
            y_train_2b[(i-1)*N_seg:i*N_seg,:]=y_tr_2b[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
            m_train_2[(i-1)*N_seg:i*N_seg,:]=m_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]#for validating
            m1_train_2[(i-1)*N_seg:i*N_seg,:]=m1_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
            gx_train_2[(i-1)*N_seg:i*N_seg,:,:]=gx_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
            
            
            x_test[(i-1)*N_seg:i*N_seg,:,:]=x_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
            y_test[(i-1)*N_seg:i*N_seg,:]=y_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
            y_testa[(i-1)*N_seg:i*N_seg,:]=y_tea[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
            y_testb[(i-1)*N_seg:i*N_seg,:]=y_teb[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
            m_test[(i-1)*N_seg:i*N_seg,:]=m_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
            gx_test[(i-1)*N_seg:i*N_seg,:,:]=gx_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]

         
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        
        # sess = tf.Session()
        if ct == k_list[0] and r == 0:
            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=tvars[:12]) 
            saver.restore(sess, "./step1/pretrained.ckpt") 
            saver1 = tf.train.Saver(max_to_keep=3)

         
        mre = 100
        
        for epoch in range(epochs):
                
            alos = 0
                
            idx = range(N_sec)
            idx = random.sample(idx,N_sec)
                
            for i in range(N_sec): # better code?
                index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
                    
                batch_x = x_train_1[index,:,:]
                batch_gx = gx_train_1[index,:,:]
                batch_x1 = x_train_2[index,:,:]
                batch_gx1 = gx_train_2[index,:,:]
                batch_y = y_train_1[index,:]
                batch_y1 = y_train_2[index,:] ###validating
                batch_ya = y_train_1a[index,:,:]
                batch_yb = y_train_1b[index,:,:]
                batch_m = m_train_1[index,:]
                batch_m1 = m1_train_2[index,:] ###validating
                    
                    
                if np.sum(batch_m)>0:
                    _, los = sess.run(
                        [train_op, cost1],
                        feed_dict = {
                            x_tr1: batch_x,
                            gx_tr1:batch_gx,
                            x_tr2: batch_x1,
                            gx_tr2:batch_gx1,
                            y_tr1: batch_y,
                            y_tr2: batch_y1, ###validating
                            y_val1: batch_ya,
                            y_val2: batch_yb,
                            m_tr1: batch_m,
                            m_tr2: batch_m1 ###validating
                                    
                    })
                    alos += los  
            print('Epoch '+str(epoch)+' station No.' + str(ct) +' batch 1: loss '+"{:.4f}".format(alos/N_sec) )
            
            

            prd_te = np.zeros([N_sec*N_seg,n_steps,1])
            
            for i in range(N_sec): # better code?
                index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
                    
                batch_x = x_test[index,:,:]
                batch_gx = gx_test[index,:,:]
                batch_x1 = x_test[index,:,:]
                batch_gx1 = gx_test[index,:,:]
                batch_y = y_test[index,:]
                batch_y1 = y_test[index,:] ###validating
                batch_ya = y_testa[index,:]
                batch_yb = y_testb[index,:]
                batch_m = m_test[index,:]
                batch_m1 = m_test[index,:] ###validating

                
                batch_prd, wf,bf = sess.run(
                    [pred_tr,W1,b1],
                    feed_dict = {
                        x_tr1: batch_x,
                        gx_tr1:batch_gx,
                        x_tr2: batch_x1,
                        gx_tr2:batch_gx1,
                        y_tr1: batch_y,
                        y_tr2: batch_y1, ###validating
                        y_val1: batch_ya,
                        y_val2: batch_yb,
                        m_tr1: batch_m,
                        m_tr2: batch_m1###validating

                                
                })
                prd_te[index,:,:]=batch_prd
            prd_o = np.zeros([N_seg,3600])
            prd_o[:,:n_steps] = prd_te[0:N_seg,:,0]
            for j in range(N_sec-1):   # 18*125    +250 = 2500
                st_idx = n_steps-(int((j+1)*n_steps/2)-int(j*n_steps/2))
                prd_o[:, n_steps+int(j*n_steps/2):n_steps+int((j+1)*n_steps/2)] = prd_te[(j+1)*N_seg:(j+2)*N_seg,st_idx:,0]
            
            po = np.reshape(prd_o,[-1])
            ye = np.reshape(y_te,[-1])
            me = np.reshape(m_te,[-1])
            rmse = np.sqrt(np.sum(np.square((po-ye)*me))/np.sum(me))
            
            print( 'Test RMSE: '+'station No.' + str(ct) +" {:.4f}".format(rmse))
                    
    

                
        print('model saving...'+ str(ct))
        save_path = saver1.save(sess, "./step2/pretain" + str(ct) + ".ckpt")
        print("Model saved in path: %s" % save_path)
        
        
save_path = saver1.save(sess, "./step2/pretrain.ckpt")
print("Model saved in path: %s" % save_path)
np.save('./step2/pred_pretain.npy',pred_all) 