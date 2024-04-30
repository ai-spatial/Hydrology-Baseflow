
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

tf.reset_default_graph()
random.seed(9001)


''' Declare constants '''
learning_rate = 0.002#0.001#0.005/3/2 default 0.002#0.0009
epochs =  60
input_size = 30
input_size_dy = input_size - 10
state_size = 30 
n_classes = 1
cv_idx = 2
N_seg = 60#60 
kb=1.0


npic = 10
n_steps = int(3600/npic)
N_sec = (npic-1)*2+1


#setting up gage dataset dimemsion reduction
input_size_gage = 60
output_size_gage = 10

''' Build Graph '''
x_tr = tf.placeholder("float", [None, n_steps,input_size_dy]) 
gx_tr = tf.placeholder("float", [None, n_steps,input_size_gage]) 

y_tr1 = tf.placeholder("float", [None, n_steps]) 
m_tr1 = tf.placeholder("float", [None, n_steps])

# y_tr2 = tf.placeholder("float", [None, n_steps]) 
# m_tr2 = tf.placeholder("float", [None, n_steps])


y_val1 = tf.placeholder("float", [None, n_steps]) 
m_val1 = tf.placeholder("float", [None, n_steps])


y_val2 = tf.placeholder("float", [None, n_steps]) 
m_val2 = tf.placeholder("float", [None, n_steps])


W1 = tf.get_variable('W_1',[input_size_gage, output_size_gage], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))#default 0.02
b1 = tf.get_variable('b_1',[output_size_gage],  tf.float32,
                                  initializer=tf.constant_initializer(0.0)) #default 0


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


w_fin = tf.get_variable('w_fin',[state_size, n_classes], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))
b_fin = tf.get_variable('b_fin',[n_classes],  tf.float32,
                                  initializer=tf.constant_initializer(0.0))



def forward(xi, gx, W1,b1,Wi,Wf,Wo,Wg,Ui,Uf,Uo,Ug,w_fin,b_fin,reuse=False):
    gx1 = tf.sigmoid(tf.matmul(gx,W1)+b1)
    x = tf.concat([xi, gx1], 2) 
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
            
    for t in range(1,n_steps):
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


h_tr,pred_tr = forward(x_tr, gx_tr, W1,b1,Wi,Wf,Wo,Wg,Ui,Uf,Uo,Ug,w_fin,b_fin)

cost = loss_measure(pred_tr,y_tr1,m_tr1) 

tvars = tf.trainable_variables()
for i in tvars:
    print(i)
    
    
grads = tf.gradients(cost, tvars)


optimizer = tf.train.AdamOptimizer(learning_rate)

train_op = optimizer.apply_gradients(zip(grads, tvars))


''' Load data '''
feat = np.load('./data/NCA_processing.npy')
label = np.load('./data/station_log.npy')
mask = np.load('./data/synthetic_mask.npy')
mask1 = np.load('./data/synthetic_mask.npy')
mask1[:,0:3600] = 0



labela = np.load('./data/simulated_eq1.npy')
labelb = np.load('./data/simulated_eq2.npy')



maska =  np.load('./data/synthetic_mask.npy')
maskb =  np.load('./data/synthetic_mask.npy')

maska1 =  np.load('./data/synthetic_mask.npy')
maskb1 =  np.load('./data/synthetic_mask.npy')


labela = np.log(labela)
labelb = np.log(labelb)

labela = labela*maska
labelb = labelb*maskb



# np.save('./data/gage_processing.npy',gage_feat)
gage_feat = np.load('./data/gage_processing.npy')


x_te = feat[:,cv_idx*3600:(cv_idx+1)*3600,:]
gx_te = gage_feat[:,cv_idx*3600:(cv_idx+1)*3600,:]
y_te = label[:,cv_idx*3600:(cv_idx+1)*3600]
y_tea = labela[:,cv_idx*3600:(cv_idx+1)*3600]
y_teb = labelb[:,cv_idx*3600:(cv_idx+1)*3600]
m_te = mask[:,cv_idx*3600:(cv_idx+1)*3600]
m_tea = maska[:,cv_idx*3600:(cv_idx+1)*3600]
m_teb = maskb[:,cv_idx*3600:(cv_idx+1)*3600]


# np.save('./results/obs_60st.npy',y_te)
if cv_idx==2:
    x_tr_1 = feat[:,:3600,:]
    y_tr_1 = label[:,:3600]
    y_tr_1a = labela[:,:3600]
    y_tr_1b = labelb[:,:3600]
    m_tr_1 = mask[:,:3600]
    m1_tr_1 = mask1[:,:3600]
    m_tr_1a = maska[:,:3600]
    m_tr_1b = maskb[:,:3600]
    gx_tr_1 = gage_feat[:,:3600,:]
    
    x_tr_2 = feat[:,3600:2*3600,:]
    y_tr_2 = label[:,3600:2*3600:]
    y_tr_2a = labela[:,3600:2*3600:]
    y_tr_2b = labelb[:,3600:2*3600:]
    m_tr_2 = mask[:,3600:2*3600:]#for validating
    m1_tr_2 = mask1[:,3600:2*3600:]
    m_tr_2a = maska[:,3600:2*3600:]
    m1_tr_2a = maska1[:,3600:2*3600:]
    m_tr_2b = maskb[:,3600:2*3600:]
    m1_tr_2b = maskb1[:,3600:2*3600:]
    gx_tr_2 = gage_feat[:,3600:2*3600,:]
    
    
x_train_1 = np.zeros([N_seg*N_sec,n_steps,input_size_dy])
y_train_1 = np.zeros([N_seg*N_sec,n_steps])
y_train_1a = np.zeros([N_seg*N_sec,n_steps])
y_train_1b = np.zeros([N_seg*N_sec,n_steps])
m_train_1 = np.zeros([N_seg*N_sec,n_steps])
m1_train_1 = np.zeros([N_seg*N_sec,n_steps])
m_train_1a = np.zeros([N_seg*N_sec,n_steps])
m_train_1b = np.zeros([N_seg*N_sec,n_steps])
gx_train_1 = np.zeros([N_seg*N_sec,n_steps,input_size_gage])

x_train_2 = np.zeros([N_seg*N_sec,n_steps,input_size_dy])
y_train_2 = np.zeros([N_seg*N_sec,n_steps])
y_train_2a = np.zeros([N_seg*N_sec,n_steps])
y_train_2b = np.zeros([N_seg*N_sec,n_steps])
m_train_2 = np.zeros([N_seg*N_sec,n_steps])#for validating
m1_train_2 = np.zeros([N_seg*N_sec,n_steps])
m_train_2a = np.zeros([N_seg*N_sec,n_steps])
m_train_2b = np.zeros([N_seg*N_sec,n_steps])
m1_train_2a = np.zeros([N_seg*N_sec,n_steps])
m1_train_2b = np.zeros([N_seg*N_sec,n_steps])
gx_train_2 = np.zeros([N_seg*N_sec,n_steps,input_size_gage])

x_test = np.zeros([N_seg*N_sec,n_steps,input_size_dy])
y_test = np.zeros([N_seg*N_sec,n_steps])
y_testa = np.zeros([N_seg*N_sec,n_steps])
y_testb = np.zeros([N_seg*N_sec,n_steps])
m_test = np.zeros([N_seg*N_sec,n_steps])
m_testa = np.zeros([N_seg*N_sec,n_steps])
m_testb = np.zeros([N_seg*N_sec,n_steps])
gx_test = np.zeros([N_seg*N_sec,n_steps,input_size_gage])



for i in range(1,N_sec+1):
    x_train_1[(i-1)*N_seg:i*N_seg,:,:]=x_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    y_train_1[(i-1)*N_seg:i*N_seg,:]=y_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    y_train_1a[(i-1)*N_seg:i*N_seg,:]=y_tr_1a[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    y_train_1b[(i-1)*N_seg:i*N_seg,:]=y_tr_1b[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_train_1[(i-1)*N_seg:i*N_seg,:]=m_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m1_train_1[(i-1)*N_seg:i*N_seg,:]=m1_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_train_1a[(i-1)*N_seg:i*N_seg,:]=m_tr_1a[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_train_1b[(i-1)*N_seg:i*N_seg,:]=m_tr_1b[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    gx_train_1[(i-1)*N_seg:i*N_seg,:,:]=gx_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    
    x_train_2[(i-1)*N_seg:i*N_seg,:,:]=x_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    y_train_2[(i-1)*N_seg:i*N_seg,:]=y_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    y_train_2a[(i-1)*N_seg:i*N_seg,:]=y_tr_2a[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    y_train_2b[(i-1)*N_seg:i*N_seg,:]=y_tr_2b[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_train_2[(i-1)*N_seg:i*N_seg,:]=m_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]#for validating
    m1_train_2[(i-1)*N_seg:i*N_seg,:]=m1_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_train_2a[(i-1)*N_seg:i*N_seg,:]=m_tr_2a[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_train_2b[(i-1)*N_seg:i*N_seg,:]=m_tr_2b[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m1_train_2a[(i-1)*N_seg:i*N_seg,:]=m1_tr_2a[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m1_train_2b[(i-1)*N_seg:i*N_seg,:]=m1_tr_2b[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    gx_train_2[(i-1)*N_seg:i*N_seg,:,:]=gx_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    
    
    x_test[(i-1)*N_seg:i*N_seg,:,:]=x_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    y_test[(i-1)*N_seg:i*N_seg,:]=y_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_test[(i-1)*N_seg:i*N_seg,:]=m_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_testa[(i-1)*N_seg:i*N_seg,:]=m_tea[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_testb[(i-1)*N_seg:i*N_seg,:]=m_teb[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    gx_test[(i-1)*N_seg:i*N_seg,:,:]=gx_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    
''' Session starts '''

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=3)


mre = 100   

print("#######################")
print('pretrining the model')
for epoch in range(epochs):
        
    alos = 0

        
    idx = range(N_sec)
    idx = random.sample(idx,N_sec)
        
    for i in range(N_sec): # better code?
        index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
            
        batch_x = x_train_1[index,:,:]
        batch_gx = gx_train_1[index,:,:]
        batch_y = y_train_1[index,:]
        batch_m = m_train_1[index,:]

        if np.sum(batch_m)>0:
            _, los = sess.run(
                [train_op, cost],
                feed_dict = {
                    x_tr: batch_x,
                    gx_tr:batch_gx,
                    y_tr1: batch_y,
                    m_tr1: batch_m

                            
            })
            alos += los


            
  
    print('Epoch '+str(epoch)+' batch 1: loss '+"{:.4f}".format(alos/N_sec) )
    
    
    alos = 0

    for i in range(N_sec): # better code?
        index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
        
        batch_x = x_train_2[index,:,:]
        batch_gx = gx_train_2[index,:,:]
        batch_y = y_train_2[index,:]
        batch_m = m_train_2[index,:]

        if np.sum(batch_m)>0:
            _, los = sess.run(
                [train_op, cost],
                feed_dict = {
                    x_tr: batch_x,
                    gx_tr:batch_gx,
                    y_tr1: batch_y,
                    m_tr1: batch_m
                            
            })
            alos += los

    
    print('Epoch '+str(epoch)+' batch 2: loss '+"{:.4f}".format(alos/N_sec) )
        
    
    prd_te = np.zeros([N_sec*N_seg,n_steps,1])
    for i in range(N_sec): # better code?
        index = range(i*N_seg, (i+1)*N_seg)
            
        batch_x = x_test[index,:,:]
        batch_gx = gx_test[index,:,:]
        batch_y = y_test[index,:]
        batch_m = m_test[index,:]

            
        batch_prd, wf,bf = sess.run(
            [pred_tr,W1,b1],
            feed_dict = {
                x_tr: batch_x,
                gx_tr:batch_gx,
                y_tr1: batch_y,
     
                m_tr1: batch_m
                        
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
    

        
    print( 'Test RMSE: '+"{:.4f}".format(rmse))
        
            
print('saving...')
np.save('./step1/pred_pretraining.npy',prd_o)
np.save('./step1/ws_pretrain.npy',wf)
np.save('./step1/bs_pretrain.npy',bf)
            
save_path = saver.save(sess, "./step1/pretrained.ckpt")
print("Model saved in path: %s" % save_path)