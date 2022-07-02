import logging
import time
import torch
import numpy as np


NUM_CLASSES = 16

@torch.no_grad()
def count_confusion(loader, model):
    
    conf_coeffs = np.zeros((NUM_CLASSES, NUM_CLASSES)).astype(int)


    model.eval()
    topk = (1,)
    maxk = max(topk)
        
    for i, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()

        outputs = model(x)
        
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        
        for actual, predicted in zip(y, pred[0]):
            conf_coeffs[actual.cpu()][predicted.cpu()] += 1
    
    return conf_coeffs
            
