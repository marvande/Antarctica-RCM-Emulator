#!/usr/bin/env python3
import torch
import random as rn
import numpy as np

# Handles handling of seeds in pytorch so that reproducible

def seed_all(seed):
	if not seed:
		seed = 10
		
		#print("[ Using Seed : ", seed, " ]")
	
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	rn.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	rn.seed(worker_seed)
	