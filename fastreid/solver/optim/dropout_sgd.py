import numpy as np
import torch

from itertools import chain
from torch.optim.optimizer import Optimizer, required


class DropoutSGD(Optimizer):
    '''
    The proposed Dropout
    '''
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, cfg=None):
        self.p_prob = cfg.SOLVER.DROPOUTSGD.P_PROB
        self.inv_ex = cfg.SOLVER.DROPOUTSGD.INV_EX
        self.layer_names = cfg.SOLVER.DROPOUTSGD.LAYER_NAMES
        self.cur_layers = cfg.SOLVER.DROPOUTSGD.BEGIN_LAYER
        self.max_layers = self.cur_layers + cfg.SOLVER.DROPOUTSGD.WINDOW_SIZE

        self.open_dropout = False

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DropoutSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DropoutSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            name = group['name']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p = d_p.add(p.data, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    if self.open_dropout:
                        for layer_name in chain(*self.layer_names[self.cur_layers:self.max_layers]):
                            if layer_name in name:
                                d_p = d_p * torch.bernoulli(d_p, p=self.p_prob)
                                if self.inv_ex:
                                    d_p /= self.p_prob
                                break
                    p.add_(d_p, alpha=-lr)

        return loss
