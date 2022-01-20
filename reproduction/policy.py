import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """Policy network, i.e., RNN controller that generates the different childNet architectures."""

    def __init__(self, batch_size, n_outputs, layer_limit):
        super(PolicyNet, self).__init__()
        
        # parameters
        
        # maximal layers limited by episodes
        self.layer_limit = layer_limit
        self.gamma = 1.0
        # number of hidden units of the controller
        self.n_hidden = 24 
        # number of outputs = number of possible hidden units of the childnet + number of possible activation functions of the childnet
        self.n_outputs = n_outputs 
        self.learning_rate = 1e-2
        self.batch_size = batch_size
        
        # Neural Network
        self.lstm = nn.LSTMCell(self.n_outputs, self.n_hidden)
        # topology fixed as a linear architecture
        self.linear = nn.Linear(self.n_hidden, self.n_outputs)
        
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def one_hot(self, t, num_classes):
        '''One hot encoder of an action/hyperparameter that will be used as input for the next RNN iteration. '''
        out = np.zeros((t.shape[0], num_classes))
        for row, col in enumerate(t):
            out[row, col] = 1
        return out.astype('float32')

    def sample_action(self, output, training):
        '''Stochasticity of the policy, picks a random action based on the probabilities computed by the last softmax layer. '''
        if training:#when training，pick randomly
            random_array = np.random.rand(self.batch_size).reshape(self.batch_size,1)
            # sample action
            return (np.cumsum(output.detach().numpy(), axis=1) > random_array).argmax(axis=1) 
        else :#when testing, choose the highest probability
            return (output.detach().numpy()).argmax(axis=1)
                
    def forward(self, training):
        ''' Forward pass. Generates different childNet architectures (number of architectures = batch_size). '''
        outputs = []
        prob = []
        actions = np.zeros((self.batch_size, self.layer_limit))
        action = not None #initialize action to don't break the while condition 
        i = 0
        counter_nb_layers = 0
        
        h_t = torch.zeros(self.batch_size, self.n_hidden, dtype=torch.float)
        c_t = torch.zeros(self.batch_size, self.n_hidden, dtype=torch.float)
        action = torch.zeros(self.batch_size, self.n_outputs, dtype=torch.float)
        
        while counter_nb_layers<self.layer_limit: 
        # yield the network structure according to the output probability by LSTM
            h_t, c_t = self.lstm(action, (h_t, c_t))
                        
            output = F.softmax(self.linear(h_t))
            counter_nb_layers += 1
            # choose actions at next step according to output
            action = self.sample_action(output, training)

            outputs += [output]
            prob.append(output[np.arange(self.batch_size),action])
            actions[:, i] = action
            action = torch.tensor(self.one_hot(action, self.n_outputs))            
            i += 1
        # compress dimensions
        prob = torch.stack(prob, 1) 
        outputs = torch.stack(outputs, 1).squeeze(2)
        
        return prob, actions
        
    #SGD
def loss(self, action_probabilities, returns, baseline):  
        ''' Policy loss. '''
        #T is the number of hyperparameters 
        # do log seperately and then sum over
        sum_over_T = torch.sum(torch.log(action_probabilities.view(self.batch_size, -1)), axis=1)
        # minus base-line
        subs_baseline = torch.add(returns,-baseline)
        return torch.mean(torch.mul(sum_over_T, subs_baseline)) - torch.sum(torch.mul (torch.tensor(0.01) * action_probabilities, torch.log(action_probabilities.view(self.batch_size, -1))))