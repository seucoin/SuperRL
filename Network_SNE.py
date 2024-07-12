import numpy as np
import torch
import torch.nn as nn
import logging
import torch.nn.init as init
import torch.nn.functional as F

class Triple_Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(Triple_Attention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size, bias=False)

        self.att_dropout = nn.Dropout(0.2)
        self.Bilinear_att = nn.Linear(self.att_size, self.att_size, bias=False)# 100,100

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        """
        q (target_rel):  (few/b, 1, dim)
        k (nbr_rel):    (few/b, max, dim)
        v (nbr_ent):    (few/b, max, dim)
        mask:   (few/b, max)
        output:
        """
        q = q.unsqueeze(1)# (5,1,100)
        orig_q_size = q.size()# [5,1,100]
        batch_size = q.size(0)# 5

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.att_size)   #(few/b, 1, num_heads, att_size) (5,1,1,100)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.att_size)   #(few/b, max, num_heads, att_size) (5,100,1,100)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.att_size)   #(few/b, max, num_heads, att_size) (5,100,100)


        q = q.transpose(1, 2)  #(few/b, num_heads, 1, att_size) (5,1,1,100)
        k = k.transpose(1, 2).transpose(2, 3)  #(few/b, num_heads, att_size, max) (5,1,100,100)
        v = v.transpose(1, 2)  #(few/b, num_heads, max, att_size) (5,1,100,100)

        x = torch.matmul(self.Bilinear_att(q), k)# (5,1,1,100)

        x = torch.softmax(x, dim=3)   # [few/b, num_heads, 1, max] (5,1,1,100)

        x = self.att_dropout(x)     # [few/b, num_heads, 1, max] (5,1,1,100)
        x = x.matmul(v)    #(few/b, num_heads, 1, att_size) (5,1,1,100)

        x = x.transpose(1, 2).contiguous()  # (few/b, 1, num_heads, att_size) (5,1,1,100)

        x = x.view(batch_size, -1, self.num_heads * self.att_size).squeeze(1) #(few/b, dim) (5,100)
        x = self.output_layer(x)  #(few/b, dim) (5,100)

        return x

class Context_Attention(nn.Module):
    def __init__(self, dim, max_neighbor, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Context_Attention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, dim, bias=False)
        self.max_neighbor = max_neighbor
        self.Bilinear_att = nn.Linear(self.head_dim, self.head_dim, bias=False)# 100,100
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, mask=None):
        B, N, C = q.shape

        q = self.linear_q(q).view(B, N, self.num_heads, self.head_dim)   #(few/b, 1, num_heads, att_size) (5,1,1,100)
        k = self.linear_k(k).view(B, N, self.num_heads, self.head_dim)   #(few/b, max, num_heads, att_size) (5,100,1,100)
        v = self.linear_v(v).view(B, N, self.num_heads, self.head_dim)   #(few/b, max, num_heads, att_size) (5,100,100)

        q = q.transpose(1, 2)  #(few/b, num_heads, 1, att_size) (5,1,1,100)
        k = k.transpose(1, 2).transpose(2, 3)  #(few/b, num_heads, att_size, max) (5,1,100,100)
        v = v.transpose(1, 2)  #(few/b, num_heads, max, att_size) (5,1,100,100)

        attn = (self.Bilinear_att(q) @ k) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask==0, -1e9)
        
        attn = attn.mean(dim=-2).unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)

        return x.squeeze(1)

class Entity_Attention_Block(nn.Module):
    def __init__(self, embedding_size, num_heads=1):
        super(Entity_Attention_Block, self).__init__()

        self.embedding_size = embedding_size
        self.entity_attention = Triple_Attention(hidden_size=self.embedding_size, num_heads=num_heads)
        self.MLPW1 = nn.Linear(2 * self.embedding_size, 2 * self.embedding_size)# (2*100,2*100)
        self.MLPW2 = nn.Linear(2 * self.embedding_size, 2 * self.embedding_size, bias=False)
        self.layer_norm = nn.LayerNorm(2 * self.embedding_size)
        init.xavier_normal_(self.MLPW1.weight)
        init.xavier_normal_(self.MLPW2.weight)

    def forward(self, entity_left, entity_right, ent_embeds_left, ent_embeds_right, V_head, V_tail, mask=None):

        input = torch.cat([entity_left, entity_right], dim=-1)# (5,200)
        head_nn_ent_aware = self.entity_attention(q=entity_right, k=ent_embeds_left, v=V_head)# (5,100)
        tail_nn_ent_aware = self.entity_attention(q=entity_left, k=ent_embeds_right, v=V_tail) # (5,100)
        concat_pair = torch.cat([head_nn_ent_aware, tail_nn_ent_aware], dim=-1)# (5,200)
        context = torch.relu(self.MLPW1(concat_pair))# (5,200)
        context_TL = self.MLPW2(context)# (5,200)
        output = self.layer_norm(context_TL + input)# (5,200)
        emb_left, emb_right = torch.split(output, self.embedding_size, dim=-1)  # (few/b, dim) 根据双线性层语义，各为(5,100)

        return emb_left, emb_right, context_TL

class Context_Attention_Block(nn.Module):
    def __init__(self, args, embedding_size, num_heads=1):
        super(Context_Attention_Block, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.context_attention = Context_Attention(self.embedding_size, self.args.max_neighbor, num_heads=num_heads, qkv_bias=False, attn_drop=0.2, proj_drop=0.2) #
        self.MLPW1 = nn.Linear(2 * self.embedding_size, 2 * self.embedding_size)# (2*100,2*100)
        self.MLPW2 = nn.Linear(2 * self.embedding_size, 2 * self.embedding_size, bias=False)
        self.layer_norm = nn.LayerNorm(2 * self.embedding_size)
        init.xavier_normal_(self.MLPW1.weight)
        init.xavier_normal_(self.MLPW2.weight)

    def forward(self, entity_left, entity_right, ent_embeds_left, ent_embeds_right, V_head, V_tail, mask=None, resdual=True):
        mask_left, mask_right = mask
        context_l = self.context_attention(q=ent_embeds_right, k=ent_embeds_left, v=V_head, mask=mask_left)
        context_r = self.context_attention(q=ent_embeds_left, k=ent_embeds_right, v=V_tail, mask=mask_right)
        concat_pair = torch.cat([context_l, context_r], dim=-1)# (5,200)
        context = torch.relu(self.MLPW1(concat_pair))# (5,200)
        context_CL = self.MLPW2(context)# (5,200)
        if(resdual):
            input = torch.cat([entity_left, entity_right], dim=-1)# (5,200)
            output = self.layer_norm(context_CL + input)# (5,200)
        else:
            output = self.layer_norm(context_CL)
        emb_left, emb_right = torch.split(output, self.embedding_size, dim=-1)  # (few/b, dim)
        return emb_left, emb_right, context_CL

class Transformer(nn.Module):
    def __init__(self, args, embedding_size, num_heads=1, num_layers=3):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        if num_layers != 0:
            self.entity_pair_attention_modules = nn.ModuleList([Entity_Attention_Block(self.embedding_size, num_heads=num_heads) for _ in range(self.num_layers)])
            self.context_pair_attention_modules = nn.ModuleList([Context_Attention_Block(args, self.embedding_size, num_heads=num_heads) for _ in range(self.num_layers)])
        self.context_TL_encoder = Entity_Attention_Block(self.embedding_size, num_heads=num_heads)
        self.context_CL_encoder = Context_Attention_Block(args, self.embedding_size, num_heads=num_heads)
    
    def forward(self, entity_left, entity_right, ent_embeds_left, ent_embeds_right, V_head, V_tail, mask=None, resdual=True):
        if self.num_layers==0:
            enhanced_left = entity_left
            enhanced_right = entity_right
        else:
            enhanced_left, enhanced_right, _ = self.context_pair_attention_modules[0](entity_left=entity_left, entity_right=entity_right,
                                                                            ent_embeds_left=ent_embeds_left, ent_embeds_right=ent_embeds_right,
                                                                            V_head=V_head, V_tail=V_tail, mask=mask, resdual=resdual)
            enhanced_left, enhanced_right, _ = self.entity_pair_attention_modules[0](entity_left=enhanced_left, entity_right=enhanced_right,
                                                                            ent_embeds_left=ent_embeds_left, ent_embeds_right=ent_embeds_right,
                                                                            V_head=V_head, V_tail=V_tail)
            for i in range(self.num_layers - 1):
                enhanced_left, enhanced_right, _ = self.context_pair_attention_modules[i+1](entity_left=enhanced_left, entity_right=enhanced_right,
                                                                                ent_embeds_left=ent_embeds_left, ent_embeds_right=ent_embeds_right,
                                                                                V_head=V_head, V_tail=V_tail, mask=mask, resdual=resdual)

                enhanced_left, enhanced_right, _ = self.entity_pair_attention_modules[i+1](entity_left=enhanced_left, entity_right=enhanced_right,
                                                                                ent_embeds_left=ent_embeds_left, ent_embeds_right=ent_embeds_right,
                                                                                V_head=V_head, V_tail=V_tail)
                                                                              
        left_CL,right_CL,context_CL = self.context_CL_encoder(entity_left=enhanced_left, entity_right=enhanced_right,
                                                                              ent_embeds_left=ent_embeds_left, ent_embeds_right=ent_embeds_right,
                                                                              V_head=V_head, V_tail=V_tail, mask=mask, resdual=resdual)
        left_TL,right_TL,context_TL = self.context_TL_encoder(entity_left=enhanced_left, entity_right=enhanced_right,
                                                                              ent_embeds_left=ent_embeds_left, ent_embeds_right=ent_embeds_right,
                                                                              V_head=V_head, V_tail=V_tail)
        context_CL = torch.cat([left_CL,right_CL], dim=-1)
        context_TL = torch.cat([left_TL,right_TL], dim=-1)

        return enhanced_left, enhanced_right, context_CL, context_TL

class Attention_Module(nn.Module):
    def __init__(self, args, embed, num_symbols, embedding_size, use_pretrain=True, finetune=True, dropout_rate=0.3, num_layers=6):
        super(Attention_Module, self).__init__()

        self.args = args
        self.embedding_size = embedding_size
        self.pad_idx = num_symbols
        self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(args.device)

        self.symbol_emb = nn.Embedding(num_symbols+1, self.embedding_size, padding_idx=self.pad_idx)
        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)

        self.gate_w = nn.Linear(2*self.embedding_size, self.embedding_size) # (2*100, 100)

        # 最终版本
        
        self.Transformer = Transformer(args=self.args, embedding_size=self.embedding_size,num_heads=1,num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(2 * self.embedding_size)

        init.xavier_normal_(self.gate_w.weight)

    def SPD_attention(self, entity_left, entity_right, rel_embeds_left, rel_embeds_right, ent_embeds_left, ent_embeds_right, entity_meta):
        """
        ent: (few/b, dim)
        nn:  (few/b, max, 2*dim)
        output:  ()
        """
        V_head = torch.relu(self.gate_w(torch.cat([rel_embeds_left, ent_embeds_left], dim=-1)))# (5,100,100)
        V_tail = torch.relu(self.gate_w(torch.cat([rel_embeds_right, ent_embeds_right], dim=-1)))# (5,100,100)

        mask = self.build_context(entity_meta)

        # 最终版本1
        enhanced_left, enhanced_right, context_CL, context_TL = self.Transformer(entity_left=entity_left, entity_right=entity_right,
                                                        ent_embeds_left=ent_embeds_left, ent_embeds_right=ent_embeds_right,
                                                        V_head=V_head, V_tail=V_tail, mask=mask, resdual=True)
        
        ent_pair_rep = torch.cat((enhanced_left, enhanced_right), dim=-1)
        # ent_pair_rep = self.layer_norm(context_CL +context_TL)
        
        return ent_pair_rep, context_CL, context_TL
    
    def build_context(self, meta):

        entity_left_connections, entity_left_degrees, entity_right_connections, entity_right_degrees = meta
        d_size = entity_left_connections.size(0)

        left_digits = torch.zeros(d_size, self.args.max_neighbor).to(self.args.device)
        right_digits = torch.zeros(d_size, self.args.max_neighbor).to(self.args.device)
        for i in range(d_size):
            left_digits[i, :entity_left_degrees[i]] = 1
            right_digits[i, :entity_right_degrees[i]] = 1
        left_digits = left_digits.reshape(-1, self.args.max_neighbor).unsqueeze(2)
        right_digits = right_digits.reshape(-1, self.args.max_neighbor).unsqueeze(2)
        
        mask_left = torch.bmm(right_digits, left_digits.transpose(1,2))
        mask_left = mask_left.reshape(-1, self.args.max_neighbor, self.args.max_neighbor)

        mask_right = torch.bmm(left_digits, right_digits.transpose(1,2))
        mask_right = mask_right.reshape(-1, self.args.max_neighbor, self.args.max_neighbor)
        
        return mask_left, mask_right


    def forward(self, entity_pairs, entity_meta): 

        entity = self.dropout(self.symbol_emb(entity_pairs))  # (few/b, 2, dim) (5,2,100)
        entity_left, entity_right = torch.split(entity, 1, dim=1)  # (few/b, 1, dim) (5,1,100) (5,1,100)
        entity_left = entity_left.squeeze(1)    # (few/b, dim) (5,100)
        entity_right = entity_right.squeeze(1)   # (few/b, dim) (5,100)

        anchor_query = torch.cat([entity_left,entity_right], dim=-1)

        entity_left_connections, entity_left_degrees, entity_right_connections, entity_right_degrees = entity_meta

        relations_left = entity_left_connections[:, :, 0].squeeze(-1)# (5,100)
        entities_left = entity_left_connections[:, :, 1].squeeze(-1)# (5,100)
        rel_embeds_left = self.dropout(self.symbol_emb(relations_left))  # (few/b, max, dim) (5,100,100)
        ent_embeds_left = self.dropout(self.symbol_emb(entities_left))   # (few/b, max, dim) (5,100,100)

        relations_right = entity_right_connections[:, :, 0].squeeze(-1)# (5,100)
        entities_right = entity_right_connections[:, :, 1].squeeze(-1)# (5,100)
        rel_embeds_right = self.dropout(self.symbol_emb(relations_right))  # (few/b, max, dim) (5,100,100)
        ent_embeds_right = self.dropout(self.symbol_emb(entities_right)) # (few/b, max, dim) (5,100,100)

        ent_pair_rep, context_CL, context_TL = self.SPD_attention(entity_left, entity_right, rel_embeds_left, rel_embeds_right, ent_embeds_left, ent_embeds_right, entity_meta) #(512,200)

        return ent_pair_rep, context_CL, context_TL, anchor_query

