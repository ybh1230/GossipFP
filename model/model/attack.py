import torch 
import torch.nn as nn 
import torch.nn.functional as F
from functools import partial
import random


def KL(logit1,logit2,reverse=False):
    if reverse:
        logit1, logit2 = logit2, logit1
    p1 = logit1.softmax(1)
    logp1 = logit1.log_softmax(1)
    logp2 = logit2.log_softmax(1) 
    return (p1*(logp1-logp2)).sum(1)

def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status

to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')


def attack(fts_clean, label, classifier, final, gossip_bank, cfg, epsilon):
    
    classifier.apply(to_adv_status)

    fts_ori = fts_clean.detach()
    if cfg['adv']['eps_rand_init']:
        noise = torch.empty_like(fts_ori).uniform_(-epsilon, epsilon)
    elif cfg['adv']['zero_init']:
        noise = torch.zeros_like(fts_ori)
    elif cfg['adv']['tiny_rand_init']:
        noise = torch.rand_like(fts_ori).sub(0.5) 
        noise = noise * 1e-6

    # Set gradient
    noise.requires_grad_()
    fts_ori.requires_grad_()
    
    fts_pt = fts_ori + noise
    fts_final = classifier(fts_pt)
    final_loss = gossip_bank.attack_loss(fts_final, label)

    grad = torch.autograd.grad(final_loss, noise, retain_graph=False, create_graph=False)[0]
    scale = gossip_bank.perturbation_scale(label, fts_ori.shape[-2:]).to(grad.dtype)
    noise = epsilon * scale * torch.nn.functional.normalize(grad.detach(), dim=1, p=2, eps=1e-6)
    
    classifier.apply(to_mix_status)

    return noise
