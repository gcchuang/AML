# Python 3.6, Numpy 1.16.2, OpenCV, scikit-image, tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import vgg19_fastmal as vgg19
import utils
from sklearn.model_selection import train_test_split
import os
import csv
import numpy as np
import pandas as pd
from random import random, shuffle

def majority_voting(votes):
    return max(votes,key=votes.count)

# make the training and testing csv file
# read the csv file as a dataframe
data = pd.read_csv("../../lib/NPM1_slide_patch_num.csv",header=None)
# set the first row as the column name, and drop the first row
data.columns = data.iloc[0]
data = data.drop(data.index[0])
# # random select 20% of the data replace the data in the dataframe
# data = data.sample(frac=0.2,replace=True)
data = data.reset_index(drop=True)
# add A to the first column
data.iloc[:,0] = 'A' + data.iloc[:,0].astype(str)
# split the dataframe into training and testing by 80% and 20%
df_train , df_test = train_test_split (data,test_size=0.2,random_state=100)
# df_train

csv_slide_ids = df_train['Slide'].tolist()
csv_labels =np.array(df_train['Target']).astype(np.uint8)
onehot_labels = np.zeros(shape=(csv_labels.shape[0], 2), dtype=np.float32)
onehot_labels[csv_labels==0,0]=1
onehot_labels[csv_labels==1,1]=1
# print(len(csv_slide_ids))

test_csv_slide_ids = df_test['Slide'].tolist()
# print(len(test_csv_slide_ids))
test_csv_labels =np.array(df_test['Target']).astype(np.uint8)

#load the data path
dataset = "/home/exon_storage1/aml_slide/single_cell_image/"
dataset2 = None
subdirsAll = os.listdir(dataset)

selected_slides = []
selected_labels=[]
selected_onehot_labels = []

test_slides = []
test_labels=[]
test_onehot_labels = []

for index, slide in enumerate(csv_slide_ids):
    if slide in subdirsAll:
        # print(f'{slide} has label {csv_labels[index]} or {onehot_labels[index,:]}')
        slide_path = os.path.join(dataset, slide)
        selected_slides.append(slide_path) 
        selected_onehot_labels.append(onehot_labels[index,:])
        selected_labels.append(csv_labels[index])

for index, slide in enumerate(test_csv_slide_ids):
    if slide in subdirsAll:
        slide_path = os.path.join(dataset, slide)
        # print(slide, ' has label ', test_csv_labels[index])
        test_slides.append(slide_path) 
        test_labels.append(test_csv_labels[index])  

selected_labels=np.array(selected_labels)
positive_ids = np.nonzero(selected_labels)[0]
negative_ids = np.nonzero(1-selected_labels)[0]

# print(selected_slides[0])
print("Negative train:", np.sum(1-selected_labels))
print("Positive train:", np.sum(selected_labels))

print("Negative test:", np.sum(1-np.array(test_labels)))
print("Positive test:", np.sum(np.array(test_labels)))

IMSIZE = 128
num_labels=2
num_steps = 10000
batch_size= 1
rpt_interval=100
min_nb_images = 1
save_dir = "../output/"
with tf.Session() as sess:
    images = tf.placeholder(tf.float32, [None, IMSIZE, IMSIZE, 3])
    true_out = tf.placeholder(tf.float32, [None, 2])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19("../model/vgg19.npy", imsize=IMSIZE)
    vgg.build_avg_pool(images, train_mode=train_mode)
    focal_loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=vgg.new_fc8, labels=true_out)) 
    train = tf.train.GradientDescentOptimizer(0.0003).minimize(focal_loss)
    sess.run(tf.global_variables_initializer())
    step_pos=0
    step_neg=0
    for step in range(1, num_steps+1):
        try:  
            offset = (step * batch_size) % (len(selected_onehot_labels) - batch_size)
            
            if step%2==0:
                sl_id = positive_ids[step_pos % (positive_ids.shape[0])]
                step_pos=step_pos+1
            else:
                sl_id = negative_ids[step_neg % (negative_ids.shape[0])]
                step_neg=step_neg+1

            slides=utils.load_folder_random(selected_slides[sl_id], max_no_img=100, crop_size=IMSIZE)
            if len(slides)>min_nb_images:
                slides = np.array(slides)
                labels = np.reshape(selected_onehot_labels[sl_id], (batch_size, num_labels))
                _,l = sess.run([train, focal_loss], feed_dict={images: slides, true_out: labels, train_mode: True})
            if (step % rpt_interval == 0):
                print('Minibatch loss at step %d: %f' % (step, l))     
            if (step % 5000 == 0):   
                save_path = os.path.join(save_dir,"max_pool_vgg19_model.npy")
                vgg.save_npy(sess, save_path)
                print("Model saved in file: %s" % save_path)    
        except IOError as e:
            print('Could not read:', selected_slides[offset], ':', e, '- it\'s ok, skipping.')        

# test classification again, should have a higher probability about tiger
    prediction_csv = dataset.split(os.path.sep)[-2]+'_mv_all_normal_max_pool2.csv' 
    header=['Slide-Id', 'True', 'Predicted']
    predictionFile= open(os.path.join("../output/", prediction_csv),'w')  
    wr = csv.writer(predictionFile, dialect='excel')
    wr.writerow(header)    
    predicted_classif = np.zeros(len(test_slides))
    true_classif = np.zeros(len(test_slides))
    for tt in range(len(test_slides)):
        
        classify_voting=[]
        probs=[]
        for run in range(1):
        #tslides = utils.load_folder(test_slides[tt], crop_size=64)
            tslides=utils.load_folder_random(test_slides[tt], max_no_img=4, crop_size=IMSIZE)
        
            tslide_id = test_slides[tt].split(os.path.sep)[-1]
            malaria_classif=0
            if len(tslides)>min_nb_images:
                tslides = np.array(tslides)

                prob = sess.run(vgg.new_prob, feed_dict={images: tslides, train_mode: False})
                #print(prob)
                classify_voting.append(np.argmax(prob))
                probs.append(prob[0,1])
        sel_id = list(test_csv_slide_ids).index(tslide_id)
        true_classif[tt] = test_csv_labels[sel_id]
        predicted_classif[tt]=majority_voting(classify_voting)
        #predicted_classif[tt]=max(classify_voting)
        wr.writerow([tslide_id, true_classif[tt], predicted_classif[tt], np.mean(probs)])  
        print(tslide_id, true_classif[tt], predicted_classif[tt], np.mean(probs))
        
        
        
    

    overall_accuracy=np.mean(true_classif==predicted_classif)    
    print('Overall accuracy', overall_accuracy)
    true_pos = true_classif[true_classif==1]
    pred_pos = predicted_classif[true_classif==1]
    positive_accuracy=np.mean(true_pos==pred_pos)
    print('Postive accuracy', positive_accuracy)
    true_neg = true_classif[true_classif==0]    
    pred_neg = predicted_classif[true_classif==0]
    negative_accuracy=np.mean(true_neg==pred_neg)
    print('Negative accuracy', negative_accuracy)   


