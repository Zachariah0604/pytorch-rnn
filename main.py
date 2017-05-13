import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from calculate import *
import torch
import torch.nn as nn
import torch.utils.data as td
from torch.autograd import Variable
from model import *

# Hyper Parameters
sequence_length = 1
input_size = 214
output_size = 1
hidden_size= 100
num_layers=1
num_classes=1
num_epochs = 5
batch_size = 1000
learning_rate = 0.01

print('..loading data')
#datas,labels=load_svmlight_file('data//data_embeding.libsvm')
datas=pd.read_csv('data//data_embeding.csv')
print('DataShape: '+str(datas.shape))
labels=datas['Intensity'].values
peptide=datas['Peptide'].values

print(datas.Number.values)
split_dot=205149


train_data=datas.head(split_dot)
print('Train dataset: '+str(train_data.Number.values))
train_label=labels[:split_dot]
test_data=datas.loc[split_dot:datas.shape[0]]
print('Test dataset: '+str(test_data.Number.values))
test_label=labels[split_dot:]

test_peptide=test_data.Peptide.values
test_idx=test_data.Number.values
print(test_idx)
#peptide=datas[:,1].toarray()

person_list=[]
merge_list=[]
test_merge_list=[]

print('..merge list according peptide') 
merge_list.append(get_merge_list(train_data))
test_merge_list.append(get_merge_list(test_data))


print('amount of peptide in training data:'+str(len(merge_list[0])))
train_data=train_data.drop(['Number','Peptide','Intensity'],axis=1)
test_data=test_data.drop(['Number','Peptide','Intensity'],axis=1)
train_data=np.array(train_data)
test_data=np.array(test_data)

def get_train_loader(number):
    list=[]
    list2=[]
    list3=[]
    for i in number:
        list.append(train_data[i])
        list2.append(train_label[i])
    list3.append(list)
    list3.append(list2)
    return list3
def get_split_list(array_list):
    list=[]
    for m in array_list:
        for n in range(len(merge_list[0][m])):
            list.append((merge_list[0][m][n][0]-1).astype(int))
    return list
print('..batch the peptide in training data')
idx=[x for x in range(len(merge_list[0]))]
idx_loader=td.DataLoader(dataset=idx,batch_size=batch_size,shuffle=True)
train_loaders=[]
print(idx_loader)
print('..split the merge list')
for index in idx_loader:
    train_index=get_split_list(index.tolist())
    train_loader=get_train_loader(train_index)
    train_loaders.append(train_loader)
h_state = None      # for initial hidden state


model = RNN(input_size, hidden_size, num_layers, num_classes) 
# Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # optimize all cnn parameters
loss_func = nn.MSELoss()
print('..training model')
for epoch in range(num_epochs):
    for step,(train,label) in enumerate(train_loaders):
        train=torch.FloatTensor(train)
        label=torch.FloatTensor(label)
        train=Variable(train[np.newaxis,:])
        label=Variable(label[np.newaxis,:,np.newaxis])

        
        prediction,h_state= model(train,h_state)

        #print(h_state)
        #print(h_state.data)
        h_state=Variable(h_state.data)

        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if step%100 ==0:
        print ('Epoch [%d/%d], step-%d, data-size [%d] Loss: %.4f' 
                %(epoch+1, num_epochs, step+1, len(train[0]),loss.data[0]))
print('..training complete')

print('..predicting')


test_data=torch.FloatTensor(test_data)
test_data=Variable(test_data[np.newaxis,:])
test_output,h_state=model(test_data,h_states)
test_output=test_output.data.numpy()

pred=pd.DataFrame({"Number":test_idx,"Peptide":test_peptide,"Intensity":test_output.reshape(1,len(test_output[0]))[0]})
print('..merge pred according peptide')
test_merge_list.append(get_merge_list(pred))
pred.to_csv('data/pred.csv')

print('calculate person coefficient..')
sum_person = 0.0
def get_pearson_x(len_of_peptide,pos,m):
    x=[]
    for j in range(len(test_merge_list[0][pos])):
            x.append(test_merge_list[m][pos][j][1])
    return x
for i in range(len(test_merge_list[0])):
    person_list.append(pearson_r(get_pearson_x(len(test_merge_list[0][i]),i,0),get_pearson_x(len(test_merge_list[1][i]),i,1)))
for i in range(len(person_list)):
    sum_person+=person_list[i]
person_mean = sum_person / float(len(person_list))
print('r= ' + str(person_mean))