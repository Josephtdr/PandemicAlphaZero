import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging
import os

log = logging.getLogger(__name__)

#Adapted from https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/NNet.py
class PandemicModel:
    """

    """
    def __init__(self, env, args):
        self.nnet = PandemicNNET(env, args)
        self.args = args

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.
        Input:
            examples: a list of training examples, where each example is of form
                      (s, pi, v, m). pi is the MCTS informed policy vector for
                      the given state, v is its value and m its mask.
        """
        batch_count = min( int(len(examples) / self.args.batch_size), 1000 )

        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
             max_lr=self.args.max_lr, epochs=self.args.n_epoch, steps_per_epoch=batch_count)

        t = tqdm(range(self.args.n_epoch), desc='Performing learning epochs')

        self.nnet.train()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        for _ in t:
            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                states, pis, vs, masks = list(zip(*[examples[i] for i in sample_ids]))
                states = torch.FloatTensor(np.array(states, dtype=np.float64))
                target_pis = torch.FloatTensor(np.array(pis, dtype=np.float64))
                target_vs = torch.FloatTensor(np.array(vs, dtype=np.float64))
                masks = torch.BoolTensor(np.array(masks, dtype=bool))
                # predict
                if self.args.cuda:
                    states, target_pis, target_vs, masks = states.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda(), masks.contiguous().cuda()

                # compute output
                out_pis, out_vs = self.nnet(states, masks)
                l_pi = self.loss_pi(target_pis, out_pis)
                l_v = self.loss_v(target_vs, out_vs)
                total_loss = l_pi + l_v
                # record loss
                pi_losses.update(l_pi.item(), states.size(0))
                v_losses.update(l_v.item(), states.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                scheduler.step()
            t.set_description(f'Performing learning: Current avg loss, {pi_losses.avg+v_losses.avg}')
                
        return pi_losses.avg, v_losses.avg

    def predict(self, state, mask):
        """
        Input:
            state: state representation vector.
        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        # preparing input
        state = torch.FloatTensor(state.astype(np.float64)).unsqueeze(0)
        mask = torch.BoolTensor(mask).unsqueeze(0)
        if self.args.cuda: 
            state = state.contiguous().cuda()
            mask = mask.contiguous().cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(state, mask)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoints',iteration=0,generation=0):
        filename=f'iter_{iteration}_gen_{generation}.pt'
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.nnet.state_dict(), filepath)

    def load_checkpoint(self, folder='checkpoints',iteration=0,generation=0):
        filename=f'iter_{iteration}_gen_{generation}.pt'
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        
        self.nnet.load_state_dict(torch.load(filepath))

class PandemicNNET(nn.Module):
    def __init__(self, env, args):        
        self.action_dim = env.action_dimensions
        self.state_dim = env.observation_dimensions

        super(PandemicNNET, self).__init__()

        self.dropout = args.dropout
        n_hidden_units = args.n_hidden_units
        fc1_dims = n_hidden_units
        fc2_dims = n_hidden_units
        fc3_dims = n_hidden_units

        self.fc1 = nn.Linear(self.state_dim, fc1_dims)
        self.fc1_bn = nn.BatchNorm1d(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc2_bn = nn.BatchNorm1d(fc2_dims)
        
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.fc3_bn = nn.BatchNorm1d(fc3_dims)

        self.fc_policy_head = nn.Linear(fc3_dims, self.action_dim)
        self.fc_value_head = nn.Linear(fc3_dims, 1)


    def forward(self, s, mask): # s: batch_size x state_dim
        s = F.dropout(F.relu(self.fc1_bn(self.fc1(s))), p=self.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc2_bn(self.fc2(s))), p=self.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc3_bn(self.fc3(s))), p=self.dropout, training=self.training)

        pi = self.fc_policy_head(s)
        v = self.fc_value_head(s)

        return masked_log_softmax(pi, mask, dim=1), torch.tanh(v)


#Taken from https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return F.log_softmax(vector, dim=dim)

#Taken from https://github.com/suragnair/alpha-zero-general/blob/master/utils.py
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
