# Sheng Wang at Feb 22 2023

import math
import logging
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from safetensors import safe_open
# from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter

from backbone.base_vit import ViT
import os
from backbone.linears import SimpleLinear
import gc
import torch.nn.utils as utils
import copy


# 设置日志级别为 INFO，输出到控制台
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s [%(levelname)s] => %(message)s",  # 设置日志格式
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)

class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class LoRA_ViT(nn.Module):
    """Applies low-rank adaptation to a vision transformer.
    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.
    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """
    def __init__(self, vit_model: ViT, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT, self).__init__()

        assert r > 0
        base_vit_dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
        dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.transformer.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            

            
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        
        return self.lora_vit(x)


class _LoRA_qkv_timm(nn.Module):
    """
    In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x)) #* self.scaling_factor
        new_v = self.linear_b_v(self.linear_a_v(x)) #* self.scaling_factor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
    
class _LoRA_qkv_timm_train(nn.Module):
    def __init__(
        self,
        qkv,  # 原始QKV线性变换层 (timm)
        linear_a_q,
        linear_b_q,
        linear_a_v,
        linear_b_v,
        task_id,  #  当前任务ID，用于加载之前任务的参数
        saved_A,  # 之前任务保存的LoRA A, B矩阵
        saved_B,
        t_layer_i,  # 第i个transformer block
        rank,
        scaling_factor,
        scaling_factor_prev,
        eval1=False,
        flag = False
    ):

        super().__init__()
        self.linear_a_q = linear_a_q.cuda()
        self.linear_b_q = linear_b_q.cuda()
        self.linear_a_v = linear_a_v.cuda()
        self.linear_b_v = linear_b_v.cuda()

        self.scaling_factor = scaling_factor.cuda()
        self.scaling_factor_prev = scaling_factor_prev.cuda()

        self.task_id = task_id
        self.qkv = qkv
        self.dim = qkv.in_features
        self.saved_A = saved_A
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank
        self.eval = eval1
        self.wrapped_param_prev1 = nn.ModuleList([ParameterWrapper(nn.Parameter(torch.Tensor([0.8]))) for _ in range(20)])
        
    def forward(self, x, flag=False):
        #print(f"当前是任务{self.task_id}"*5)
        #print(self.task_id)
        w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)
        #print("ssssssssssssssssssss"*5)
        #print("ssssssssssssssssssss"*5)
        #params = list(self.scaling_factor_prev.named_parameters())
        #print(params) 
        new_q, new_v = 0, 0
        
        if not flag:
        
            #print("trainingtrainingtrainingtrainingtrainingtraining")
            
            for i in range(self.task_id):
            # for i in range(0):
                #print(str(i)*50)
                saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
                Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
                _, (A_q, B_q) = Q
                _, (A_v, B_v) = V
    
                w_a_linear_q.weight = Parameter(A_q.weight)
                w_a_linear_q.weight.requires_grad = False 
                w_a_linear_q.to(x.device)
                w_b_linear_q.weight = Parameter(B_q.weight)
                w_b_linear_q.weight.requires_grad = False 
                w_b_linear_q.to(x.device)
                w_a_linear_v.weight = Parameter(A_v.weight)
                w_a_linear_v.weight.requires_grad = False 
                w_a_linear_v.to(x.device)
                w_b_linear_v.weight = Parameter(B_v.weight)
                w_b_linear_v.weight.requires_grad = False  
                w_b_linear_v.to(x.device)
    
                # 在这里打印 Q 和 V 的维度
                #print(f"Block {self.t_layer_i} - Q dimensions: ({w_a_linear_q.weight.shape}, {w_b_linear_q.weight.shape})")
                #print(f"Block {self.t_layer_i} - V dimensions: ({w_a_linear_v.weight.shape}, {w_b_linear_v.weight.shape})")
    
                if i ==0 :
                    #print("当任务id为0时"*5)
                    new_q = self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                    new_v = self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )
                else:
                    #print("当任务id为不为0时"*5)
                    new_q += self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                    new_v += self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )
                    
        else:
            #print("inferenceinferenceinfernenceinfernenceinfernence")     
            #print(self.task_id)
            self.task_id = 3              
            for i in range(self.task_id+1):
            
                #print(str(i)*50)
                #saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
                
                # 构建保存的 LoRA 权重文件路径
                saved_A_path = os.path.join("/media/AI4MED1/gs/SD-LoRA/logs/sdlora/bloodmnist_224/BS128_LR0.001_fEP5_lEP5/", f'lora_w_a_{i}.pt')
                saved_B_path = os.path.join("/media/AI4MED1/gs/SD-LoRA/logs/sdlora/bloodmnist_224/BS128_LR0.001_fEP5_lEP5/", f'lora_w_b_{i}.pt')
                saved_sf_path = os.path.join("/media/AI4MED1/gs/SD-LoRA/logs/sdlora/bloodmnist_224/BS128_LR0.001_fEP5_lEP5/", f'scaling_factor3.pt')

                    # 确保文件存在
                if os.path.exists(saved_A_path) and os.path.exists(saved_B_path):
                    # 加载保存的 A 和 B 权重
                    saved_A_i = torch.load(saved_A_path, map_location='cpu')
                    saved_B_i = torch.load(saved_B_path, map_location='cpu')
                    # 读取 scaling_factor 并应用
                if os.path.exists(saved_sf_path):
                    saved_sf_i = torch.load(saved_sf_path, map_location='cpu')
                    sf = saved_sf_i[3]  # 获取对应任务的 scaling_factor
                    #print(sf)
                    # 更新对应的 scaling factor
                    #print(self.scaling_factor_prev.param)
                    #print(nn.Parameter(sf)[0])
                    sf = nn.Parameter(sf)
#                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#                    print(sf)
#                for j in range(int(i)):
#                    #print(j)
#                    self.scaling_factor_prev[j].param = nn.Parameter(sf[j].clone())
#                    self.scaling_factor_prev[j].param.requires_grad = False  # Ensure that it's not trainable
                #params = list(self.scaling_factor_prev.named_parameters())
                #print(params) 
                # 打印调试信息
                #print(f"Scaling factor {i} applied with value: {sf}")
            
                Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
                _, (A_q, B_q) = Q
                _, (A_v, B_v) = V
                
                
                if i == 3:
                
                    #print("导入当前task3的lora权重导入当前task3的lora权重导入当前task3的lora权重")
                
                    self.linear_a_q.weight = Parameter(A_q.weight)
                    self.linear_a_q.weight.requires_grad = False 
                    self.linear_a_q.to(x.device)       
                    self.linear_b_q.weight = Parameter(B_q.weight)
                    self.linear_b_q.weight.requires_grad = False 
                    self.linear_b_q.to(x.device)
                    self.linear_a_v.weight = Parameter(A_v.weight)
                    self.linear_a_v.weight.requires_grad = False 
                    self.linear_a_v.to(x.device)       
                    self.linear_b_v.weight = Parameter(B_v.weight)
                    self.linear_b_v.weight.requires_grad = False 
                    self.linear_b_v.to(x.device)   
                    
                else:  
                                                                    
                    w_a_linear_q.weight = Parameter(A_q.weight)
                    w_a_linear_q.weight.requires_grad = False 
                    w_a_linear_q.to(x.device)
                    w_b_linear_q.weight = Parameter(B_q.weight)
                    w_b_linear_q.weight.requires_grad = False 
                    w_b_linear_q.to(x.device)
                    w_a_linear_v.weight = Parameter(A_v.weight)
                    w_a_linear_v.weight.requires_grad = False 
                    w_a_linear_v.to(x.device)
                    w_b_linear_v.weight = Parameter(B_v.weight)
                    w_b_linear_v.weight.requires_grad = False  
                    w_b_linear_v.to(x.device)
                
                    self.scaling_factor_prev[i].param = nn.Parameter(sf[i].clone())
                    self.scaling_factor_prev[i].param.requires_grad = False  # Ensure that it's not trainable
                    
                    params = list(self.scaling_factor_prev[i].named_parameters())
                    #print(params) 
                    
                    new_q += self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                    new_v += self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )
                
        #self.scaling_factor[0].param = nn.Parameter(sf[0].clone()) # true时开放
        #params1 = list(self.scaling_factor[0].named_parameters())
        #print(params1)
        
        
        
                
        new_q += self.scaling_factor[0]( self.linear_b_q(self.linear_a_q(x)) )
        new_v += self.scaling_factor[0]( self.linear_b_v(self.linear_a_v(x)) )
        #params2 = list(self.scaling_factor.named_parameters())
        #print(params2) 

#        for name, param in self.scaling_factor[0].named_parameters():
#            print(name)
#            print("================================================")
#            print(param)        
        
        qkv = self.qkv(x) 
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

class _LoRA_qkv_timm_eval(nn.Module):
    """
    In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(self, task_id, qkv: nn.Module, saved_A, saved_B, t_layer_i, rank, scaling_factor,  scaling_factor_prev, save_file):
        super().__init__()
        self.task_id = task_id
        self.qkv = qkv
        self.dim = qkv.in_features
        self.saved_A = saved_A
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank

        self.save_file = save_file
        self.scaling_factor = scaling_factor.cuda()
        self.scaling_factor_prev = scaling_factor_prev.cuda()


    def forward(self, x):
        device = torch.device('cuda:3')  # 目标设备
        new_q, new_v = 0, 0
        x = x.to(device)
        w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        w_a_linear_q.to(device)
        w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        w_b_linear_q.to(device)
        w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        w_a_linear_v.to(device)
        w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)
        w_b_linear_v.to(device)


        file_path = self.save_file+'scaling_factor'+str(self.task_id)+'.pt'
        scaling_param = torch.load(file_path)

        for i in range(self.task_id):
            saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
            Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
            _, (A_q, B_q) = Q
            _, (A_v, B_v) = V

            w_a_linear_q.weight = Parameter(A_q.weight)
            w_b_linear_q.weight = Parameter(B_q.weight)
            w_a_linear_v.weight = Parameter(A_v.weight)
            w_b_linear_v.weight = Parameter(B_v.weight)

            if i ==0 :
                new_q = self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                new_v = self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )
            else:
                new_q += self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                new_v += self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )
        
        #device = torch.device('cuda:3')  # 目标设备
        
        # 在进行每个操作之前检查张量的设备
        print(f"x is on device: {x.device}")
         
        output_q = w_a_linear_q(x)
        print(f"w_a_linear_q(x_cuda) is on device: {output_q.device}")
        
        output_wbq = w_b_linear_q(output_q)
        print(f"w_b_linear_q(output_q) is on device: {output_wbq.device}")
        
        new_q = self.scaling_factor.to(device)
        #print(f"new_q is on device: {new_q.device}")
        
        # Repeat for new_v
        #output_v = w_a_linear_v(x_cuda).to(x.device)
        #print(f"w_a_linear_v(x_cuda) is on device: {output_v.device}")
        
        #output_wbv = w_b_linear_v(output_v).to(x.device)
        #print(f"w_b_linear_v(output_v) is on device: {output_wbv.device}")
        
        #new_v = self.scaling_factor[0](output_wbv).to(x.device)
        #print(f"new_v is on device: {new_v.device}")




        new_q = self.scaling_factor[0]( w_b_linear_q(w_a_linear_q(x))).to(x.device)
        new_v = self.scaling_factor[0]( w_b_linear_v(w_a_linear_v(x))).to(x.device)
 
        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
    


class ParameterWrapper(nn.Module):
    def __init__(self, param):
        super(ParameterWrapper, self).__init__()
        self.param = param
    
    def forward(self, x):
        # print('x, param', x.device(), self.pram.device())
        return x * self.param
    
class MyLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyLinear, self).__init__()
        self.linear_b_q = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_b_q = utils.weight_norm(self.linear_b_q)

    def forward(self, x):
        return self.linear_b_q(x)


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, num_classes: int = 0, increment=10, filepath = './', lora_layer=None, eval=False, index=True, cur_task_index=None):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        self.rank =r
        self.base_vit = copy.deepcopy(vit_model)  # 复制原始ViT模型作为基础模型
        
        self.save_file = '/media/AI4MED1/gs/SD-LoRA/logs/sdlora/bloodmnist_224/BS128_LR0.001_fEP5_lEP20/'
        self.increment = increment
        # 在训练模式下设置保存路径和增量值
        if not eval:
            self.save_file = filepath
            self.increment = increment
            os.makedirs(self.save_file, exist_ok=True)
            # logging.info('save_file', self.save_file)

        # 确定应用LoRA的层
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))
        logging.info('Use LoRA in {}'.format(self.lora_layer))


        self.w_As, self.w_Bs = [], []  # 当前任务需要更新的LoRA矩阵 24

        # 初始化任务ID
        if index:
            logging.info('Initialize task-id and curtask id')
            self.task_id, self.cur_id = 0,0
        
        if cur_task_index != None:
            #logging.info('Update the network!!!', cur_task_index)
            self.task_id = cur_task_index

        # 冻结基础模型和输入模型额参数
        for param in self.base_vit.parameters():
            param.requires_grad = False
        for param in vit_model.parameters():
            param.requires_grad = False
#        for name, param in self.base_vit.named_parameters():
#            print(name)
        # 加载之前任务的LoRA参数
        saved_lora_A, saved_lora_B = {}, {}
        for i in range(self.task_id):
            file_path = os.path.join(self.save_file, 'lora_w_a_'+str(i)+'.pt')
            saved_lora_A['saved_A_'+str(i)] = torch.load(file_path)
            file_path = os.path.join(self.save_file,'lora_w_b_'+str(i)+'.pt')
            saved_lora_B['saved_B_'+str(i)] = torch.load(file_path)
            logging.info('Load LoRA parameters from {} and {}'.format(os.path.join(self.save_file, 'lora_w_a_'+str(i)+'.pt'),
                                                                      os.path.join(self.save_file,'lora_w_b_'+str(i)+'.pt')))

        # 初始化缩放参数 (幅度可学习)
        scaling_factor = nn.Parameter(torch.Tensor([0.8]))
        self.wrapped_param = nn.ModuleList([ParameterWrapper(scaling_factor)])
        self.wrapped_param_prev = nn.ModuleList([ParameterWrapper(nn.Parameter(torch.Tensor([0.8]))) for _ in range(20)])

        # 对每个transformer块应用LoRA
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            # 创建LoRA参数并替换原始QKV注意力机制
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            if not eval:
                blk.attn.qkv = _LoRA_qkv_timm_train(
                    w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v, 
                    self.task_id, saved_lora_A, saved_lora_B, t_layer_i, self.rank , self.wrapped_param, self.wrapped_param_prev, eval1=False
                )
            else:
                blk.attn.qkv = _LoRA_qkv_timm_eval(self.task_id, w_qkv_linear, saved_lora_A, saved_lora_B, t_layer_i, self.rank, self.wrapped_param, self.wrapped_param_prev, self.save_file) 

        self.reset_parameters()
        self.lora_vit = vit_model
        if not eval:
            self.lora_vit.head = torch.nn.Identity()
        else:
            self.reset_lora_vit_head()



    def reset_lora_vit_head(self):
        task_incremental = self.increment
        self.lora_vit.head = self.generate_fc(768, (self.task_id)*task_incremental).cuda()
        print(str(self.task_id-1))
        temp_weights = torch.load(self.save_file+'CLs_weight'+str(self.task_id)+'.pt') 
        temp_bias = torch.load(self.save_file+'CLs_bias'+str(self.task_id)+'.pt') 

        self.lora_vit.head.weight.data = temp_weights.data.cuda()
        self.lora_vit.head.bias.data = temp_bias.data.cuda()


    # This part is only used during the evaluation
    def reset(self, eval=False):
        self.__init__(self.base_vit, self.rank, lora_layer=None, eval=eval, index=False)

    def reset_parameters(self) -> None:
        # if self.task_id ==0: 
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
                # nn.init.kaiming_uniform_(w_A.linear_b_q.weight, a=math.sqrt(5) )
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)


    def save_wrap_param(self, filename):
        if self.task_id ==1:   
            scaling_param = torch.zeros(20,20)
        else:
            scaling_param = torch.load(filename + '/' + 'scaling_factor'+str(self.task_id-2)+'.pt')
        i = self.task_id-1
        # 创建 scaling_param 的副本，避免原地修改
        scaling_param = scaling_param.clone()
        # print('save i', i)
        for j in range(i+1):
            if j == i:
                scaling_param[i][j] = self.wrapped_param[0].param.clone()
            else:
                scaling_param[i][j] = self.wrapped_param_prev[j].param.clone()  
        #print(scaling_param.shape)
        torch.save(scaling_param, filename + '/' + 'scaling_factor'+str(self.task_id-1)+'.pt')
        
    def save_lora_parameters(self, filename: str, task_id) -> None:
        self.task_id += 1
        torch.save(self.w_As, os.path.join(filename, 'lora_w_a_'+str(task_id)+'.pt'))
        torch.save(self.w_Bs, os.path.join(filename, 'lora_w_b_'+str(task_id)+'.pt'))
        logging.info('*** Save lora parameter to ' + os.path.join(filename, 'lora_w_a_'+str(task_id)+'.pt'))
        logging.info('*** Save lora parameter to ' + os.path.join(filename, 'lora_w_b_'+str(task_id)+'.pt'))



    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def load_eval_vit(self):
        self.lora_vit = copy.deepcopy(self.base_vit)
        saved_lora_A, saved_lora_B = {}, {}
        for i in range(self.task_id):
            file_path = os.path.join(self.save_file, 'lora_w_a_'+str(i)+'.pt')
            saved_lora_A['saved_A_'+str(i)] = torch.load(file_path)
            file_path = os.path.join(self.save_file, 'lora_w_b_'+str(i)+'.pt')
            saved_lora_B['saved_B_'+str(i)] = torch.load(file_path)
            logging.info('Load LoRA parameters from {} and {}'.format(os.path.join(self.save_file, 'lora_w_a_'+str(i)+'.pt'),
                                                                      os.path.join(self.save_file, 'lora_w_a_'+str(i)+'.pt')))

        # for param in self.eval_vit.parameters():
        for param in self.lora_vit.parameters():
            param.requires_grad = False
        
        # for t_layer_i, blk in enumerate(self.eval_vit.blocks):
        for t_layer_i, blk in enumerate(self.lora_vit.blocks):
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            blk.attn.qkv = _LoRA_qkv_timm_eval(self.task_id, w_qkv_linear, saved_lora_A, saved_lora_B, t_layer_i, self.rank)    
        self.reset_lora_vit_head()

    def _load_task_sharpness_map(self):
        """
        从 {self.save_file}/task_sharpness.json 读取 {task_id: sharpness}。
        若不存在或异常，返回空 dict。
        """
        hist_dir = getattr(self, "save_file", None)
        if hist_dir is None: 
            return {}
        path = os.path.join(hist_dir, "task_sharpness.json")
        if not os.path.exists(path):
            return {}
        try:
            import json
            with open(path, "r") as f:
                data = json.load(f)
            return {int(k): float(v) for k, v in data.items()}
        except Exception:
            return {}

    def _sharpness_to_energy_ratio_map(
        self,
        scores,            # {task_id: sharpness}
        cur_task_id,
        er_min=0.30,       # 允许直接设成你想要的区间
        er_max=0.85,
        tau=None,          # 本方案不用 softmax，留参兼容
        age_decay=0.0,
    ):
        if not scores:
            return {}

        # 1) 年龄衰减
        items = []
        for t, s in scores.items():
            age = 0 if cur_task_id is None else max(0, cur_task_id - t)
            s_adj = max(0.0, float(s) - age_decay * age)
            items.append((int(t), s_adj))

        vals = torch.tensor([s for _, s in items], dtype=torch.float32)
        s_min, s_max = float(vals.min()), float(vals.max())

        # 2) 线性缩放到 [er_min, er_max]
        if s_max - s_min < 1e-12:
            # 所有 sharpness 几乎一样，就给它们均匀摊开一点（防止全都一个值）
            er_list = torch.linspace(er_min, er_max, steps=len(items)).tolist()
            return {t: float(e) for (t, _), e in zip(items, er_list)}

        er_list = er_min + (er_max - er_min) * ((vals - s_min) / (s_max - s_min))
        return {t: float(e) for (t, _), e in zip(items, er_list)}

    def compute_ortho_loss(self, energy_ratio=0.8):

        device = self.w_As[0].weight.device
        loss = torch.tensor(0.0, device=device)
        l2_loss = torch.tensor(0.0, device=device)
        num_layer = len(self.w_As)

        scores = self._load_task_sharpness_map()
        er_map = self._sharpness_to_energy_ratio_map(
            scores, cur_task_id=self.task_id, er_min=0.5, er_max=0.9, age_decay=0.0    ### blood
        ) if len(scores) > 0 else {}

        hist_dir = getattr(self, "save_file", None)

        for i in range(self.task_id):

            energy_ratio_i = float(er_map.get(i, energy_ratio))
            # energy_ratio_i = 0.1
            # print(energy_ratio_i)
            file_path_A = os.path.join(hist_dir, f"lora_w_a_{i}.pt") if hist_dir else ""
            file_path_B = os.path.join(hist_dir, f"lora_w_b_{i}.pt") if hist_dir else ""
            # print(file_path)
            if not file_path_A or not os.path.exists(file_path_A):
                continue
            prev_w_As = torch.load(file_path_A, map_location=device)
            prev_w_Bs = torch.load(file_path_B, map_location=device)
            for j in range(num_layer):
                cur_wa = self.w_As[j].weight  #  [10 x 768]
                cur_wb = self.w_Bs[j].weight  #  [768 x 10]
                Delta_cur = cur_wb @ cur_wa
                # print(cur_wa.shape)

                try:
                    prev_wa = prev_w_As[j].weight.to(cur_wa.device).detach()
                    prev_wb = prev_w_Bs[j].weight.to(cur_wb.device).detach()
                except Exception:
                    if isinstance(prev_w_As, (list, tuple)) and isinstance(prev_w_As[j], torch.Tensor):
                        prev_wa = prev_w_As[j].to(cur_wa.device).detach()
                        prev_wb = prev_w_Bs[j].to(cur_wb.device).detach()
                    else:
                        continue

                try:
                    # SVD ：prev_w ≈ U Σ V^T
                    Delta = prev_wb @ prev_wa  # [768 x 768]
                    U, S, Vh = torch.linalg.svd(Delta, full_matrices=False)   
                
                    U_r = U[:, :10]  
                    S_r = S[:10]     
                    Vh_r = Vh[:10, :] 

                except RuntimeError:
                    continue  
  
                cumulative_energy = torch.cumsum(S_r, dim=0) / torch.sum(S_r).clamp_min(1e-12)
                topk = int(torch.searchsorted(cumulative_energy, energy_ratio_i).item()) + 1
                topk = max(1, min(topk, U_r.size(1))) 
                #print(topk)
                Uk = U_r[:, :topk]                  # [768 x K] 
                Vk = Vh_r[:topk, :]                  # [K x 768] 
                D = Uk @ Vk

                sigma = S_r[:topk]                  # [topk]
                # print(sigma.shape)
                proj = torch.matmul(Uk.T, cur_wb)  # [K x 10]
                # proj = torch.matmul(D, Delta_cur.T)  # [K x 10]
                # print(proj.shape)
                temp = torch.sum(torch.abs(proj) * sigma.view(-1, 1))  
                # print(temp)
                loss = loss + temp * 10.0
                l2_loss += torch.norm(cur_wb, p=2)

        return loss, l2_loss

    def forward(self, x: Tensor, loss= False, eval=False) -> Tensor:
        if eval:
            self.reset(eval=True)
            return self.lora_vit(x)
        elif loss:
            loss, l2_loss = self.compute_ortho_loss()
            #print("ortho_loss: ", loss)
            return self.lora_vit(x), loss, l2_loss
        else:
            return self.lora_vit(x)