from sklearn.mixture import GaussianMixture
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal
import numpy as np



# Training
def train(epoch,net,net2,optimizer, input_x, input_x2, labels, w_x):
    net.train()
    net2.eval() #fix one network and train the other
    
    ## before that we need to split the data into labeled and unlabeled
    ## data.labeled
    ## data.unlabeled



    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        

def warmup(model, data, optimizer, loss_fn, train_mask, val_mask, no_val, reliability_list = None):
    model.train()
    # num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    # for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        #inputs, labels = inputs.cuda(), labels.cuda() 
    optimizer.zero_grad()
    if len(data.y.shape) != 1:
        y = data.y.squeeze(1)
    else:
        y = data.y
    outputs = model(data)               
    loss = loss_fn(outputs, labels)      
    # if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
    conf_penalty = NegEntropy()
    penalty = conf_penalty(outputs)
    L = loss + penalty      
    # elif args.noise_mode=='sym':   
    #     L = loss
    L.backward()  
    optimizer.step() 


def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


class NoiseAda(nn.Module):
    def __init__(self, class_size):
        super(NoiseAda, self).__init__()
        P = torch.FloatTensor(build_uniform_P(class_size,0.1))
        self.B = torch.nn.parameter.Parameter(torch.log(P))
    
    def forward(self, pred):
        P = F.softmax(self.B, dim=1)
        return pred @ P


def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = np.float64(noise) / np.float64(size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (np.float64(1)-np.float64(noise))*np.ones(size))
    
    diag_idx = np.arange(size)
    P[diag_idx,diag_idx] = P[diag_idx,diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


# def d_model_train(model, data, optimizer, loss_fn, train_mask, val_mask, no_val):
#     optimizer.zero_grad()
#     preds = model(data)
#     if len(data.y.shape) != 1:
#         y = data.y.squeeze(1)
#     else:
#         y = data.y
#     if loss_fn.reduction != 'none':
#         train_loss = loss_fn(preds[train_mask], y[train_mask])
#     else:
#         train_loss = (loss_fn(preds[train_mask], y[train_mask]) * reliability_list[train_mask]).mean()
#     train_loss.backward()
#     optimizer.step()
#     train_acc, _ = test(model, data, False, train_mask)
#     if not no_val:
#         val_loss = loss_fn(preds[val_mask], y[val_mask])
#         val_acc, _ = test(model, data, False, val_mask)
#     else:
#         val_loss = 0
#         val_acc = 0
#     return train_loss, val_loss, val_acc, train_acc

def cotrain_train(model1, model2, data, optimizer, loss_fn, train_mask, val_mask, no_val):
     pass


def cotrain_test(model, data, return_embeds, mask):
    pass





