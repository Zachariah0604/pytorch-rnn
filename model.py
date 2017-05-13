import torch
import torch.nn as nn
import torch.utils.data as td
from torch.autograd import Variable
# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  
    
    def forward(self, x):
        out = self.linear(x)
        return out

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
      
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.rnn=nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
            )
        self.fc=nn.Linear(hidden_size,num_classes)
    
    def forward(self,x,h_state):
        # Set initial states 
        #h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        #c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        #print(x.data.numpy().shape)
        # Forward propagate RNN
        r_out,h_state= self.rnn(x, h_state)
        #print(r_out.size(1))
        # Decode hidden state of last time step
        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.fc(r_out[:, time_step, :]))
        
        return torch.stack(outs, dim=1),h_state
class DBN(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass