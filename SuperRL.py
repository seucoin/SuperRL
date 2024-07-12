


import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable
from Network_SNE import Attention_Module
from Aggerator import *
from info_nce import *


class MRL(nn.Module):
    def __init__(self, args, num_symbols, embedding_size, embed, use_pretrain, finetune):
        super(CIAN, self).__init__()

        self.args = args
        self.entity_encoder = Attention_Module(self.args, embed=embed, num_symbols = num_symbols,
                                                embedding_size = embedding_size,
                                                use_pretrain=use_pretrain, finetune=finetune,num_layers=self.args.num_layers)


        self.Aggerator = SoftSelectPrototype(self.args)


    def forward(self, support, support_meta,
                query, query_meta,
                false = None, false_meta=None, is_train=True):


        if is_train:

            support_rep, _, _, _ = self.entity_encoder(support, support_meta)# (5,200) support(5,2) 
            query_rep, context_CL, context_TL, anchor = self.entity_encoder(query, query_meta)# (512,200)
            false_rep, context_CL_neg, context_TL_neg, _ = self.entity_encoder(false, false_meta)# (512,200)
            support_query = self.Aggerator(support_rep, query_rep) #(1, emb_dim) (512,200)
            support_false = self.Aggerator(support_rep, false_rep) #(1, emb_dim) (512,200)

            # loss_infonce = InfoNCE(negative_mode='paired')
            # loss_info_TL = loss_infonce(query_rep, context_TL, context_TL_neg.unsqueeze(1))
            # loss_info_CL = loss_infonce(query_rep, context_CL, context_CL_neg.unsqueeze(1))
            # loss_info = 0.05 * (loss_info_CL + loss_info_TL)

            # loss_infonce = InfoNCE(negative_mode='paired')
            # loss_info_TL = loss_infonce(anchor, context_TL, context_TL_neg.unsqueeze(1))
            # loss_info_CL = loss_infonce(anchor, context_CL, context_CL_neg.unsqueeze(1))
            # loss_info = 0.05 * (loss_info_CL + loss_info_TL)

            loss_infonce = InfoNCE(negative_mode='paired')
            loss_info_TL = loss_infonce(context_CL, context_TL, context_TL_neg.unsqueeze(1))
            loss_info_CL = loss_infonce(context_TL, context_CL, context_CL_neg.unsqueeze(1))
            loss_info = self.args.lamda * (loss_info_CL + loss_info_TL)

            positive_score = torch.sum(query_rep * support_query, dim=1) #size:([batch_size])
            negative_score = torch.sum(false_rep * support_false, dim=1)


        else:
            support_rep, _, _, _ = self.entity_encoder(support, support_meta)
            query_rep, context_CL, context_TL, anchor = self.entity_encoder(query, query_meta)# (512,200)
            support_query = self.Aggerator(support_rep, query_rep) #(1, emb_dim) (128,100)


            # context_CL = F.normalize(context_CL, dim=-1).unsqueeze(1)
            # context_TL = F.normalize(context_TL, dim=-1).unsqueeze(1)
            # consistency = 0.09*context_TL @ context_CL.transpose(-2,-1)
            # consistency = consistency.squeeze()
            positive_score = torch.sum(query_rep * support_query, dim=1) 
            # positive_score = positive_score+consistency
            negative_score = None
            loss_info = None
            # sim = None


        return positive_score, negative_score, loss_info
