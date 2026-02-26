import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

import timm
from backbone.lora import LoRA_ViT_timm
import torch.distributed as dist
import random
import os
import torch
torch.cuda.empty_cache()
num_workers = 8


# ===================== Sharpness-as-Loss: helpers =====================

def _get_trainable_lora_ab_params(model):
    keys = ["linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v"]
    params = []
    for n, p in model.named_parameters():
        print("11111111111111111111111111111111111")
        print(p)
        if p.requires_grad and any(k in n.lower() for k in keys):
            params.append(p)

    if len(params) == 0:
        params = [p for p in model.parameters() if p.requires_grad]
    return params

def sharpness_loss_from_ce(model, ce_loss, eps=1e-12, log_smooth=True):

    lora_params = _get_trainable_lora_ab_params(model)
    grads = torch.autograd.grad(
        ce_loss,
        lora_params,
        create_graph=True,  
        retain_graph=True,   
        allow_unused=True,
    )
    sq_terms = []
    for g in grads:
        if g is not None:
            sq_terms.append((g ** 2).sum())
    if len(sq_terms) == 0:
        return ce_loss.new_zeros(())
    s = torch.sqrt(torch.stack(sq_terms).sum() + eps)
    if log_smooth:
        s = torch.log1p(s)  
    return s

# =====================================================================


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        print("33333333333333333333333")
        print(args)
        self._network = IncrementalNet(args, True)

        self.lambda_s = args.get("lambda_s", 0.2)

        self.sharp_warmup_ep = args.get("sharp_warmup_ep", 5)

        self.sharp_every_k = args.get("sharp_every_k", 4)

        self.sharp_log_smooth = args.get("sharp_log_smooth", True)

    def _estimate_task_sharpness_sam(self, model, data_loader, rho=0.05, max_batches=10):
        net = model.module if isinstance(model, nn.DataParallel) else model
        was_training = net.training
        net.eval()

        lora_params = []
        for n, p in net.named_parameters():
            if p.requires_grad and any(k in n.lower() for k in ["linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v"]):
                lora_params.append(p)
        if len(lora_params) == 0:
      
            lora_params = [p for p in net.parameters() if p.requires_grad]

        backups = [p.data.clone() for p in lora_params]

        s_sum, n = 0.0, 0
        try:
            it = 0
            for batch in data_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    _, x, y = batch
                else:
                    x, y = batch
                x, y = x.to(self._device), y.to(self._device)

                # baseline
                for p in net.parameters():
                    if p.grad is not None:
                        p.grad = None
                logits = net(x)
                if not torch.is_tensor(logits):
                    logits = logits["logits"]
                L = F.cross_entropy(logits, y)

                # grads wrt LoRA params
                grads = torch.autograd.grad(L, lora_params, create_graph=False, retain_graph=False, allow_unused=True)

                # add perturbation
                with torch.no_grad():
                    for p, g in zip(lora_params, grads):
                        if (p is None) or (g is None):
                            continue
                        g_norm = g.norm().clamp_min(1e-12)
                        p.data.add_(rho * g / g_norm)

                # worst-case loss
                logits_adv = net(x)
                if not torch.is_tensor(logits_adv):
                    logits_adv = logits_adv["logits"]
                L_adv = F.cross_entropy(logits_adv, y)

                s_sum += float((L_adv - L).detach().cpu())
                n += 1

                # restore
                with torch.no_grad():
                    for p, b in zip(lora_params, backups):
                        p.data.copy_(b)

                it += 1
                if it >= max_batches:
                    break
        finally:
            if was_training:
                net.train()
        return float(s_sum / max(n, 1))

    def _save_task_sharpness(self, score: float):
        save_dir = self.args["filepath"]
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "task_sharpness.json")
        data = {}
        if os.path.exists(path):
            try:
                import json
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data[str(self._cur_task)] = float(score)
        with open(path, "w") as f:
            import json
            json.dump(data, f, indent=2)
    ###########################################################################

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        print("cls: ", self._total_classes)
        self._network.update_fc(self._total_classes)  # !!!!!
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=[self._device], output_device=self._device, find_unused_parameters=True)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

        self._train(self.train_loader, self.test_loader)

        # to test
        # self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def update_network(self, index=True):
        # if use VIT-B-16
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)

        # if use DINO
        # model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=0)

        # SD-LoRA-RR
        '''
        if self._cur_task >=4 and self._cur_task <8:
            rank = 8 #8
        elif self._cur_task >=8:
            rank = 6 #6
        # elif self._cur_task >=8:
        #     rank = 4
        else:
            rank = 10
        '''
        rank=10

        model = LoRA_ViT_timm(vit_model=model.eval(), r=rank, num_classes=10, index=index, 
                              increment= self.args['increment'], filepath=self.args['filepath'], 
                              cur_task_index= self._cur_task
                              )

        # model = LoRA_ViT_timm(vit_model=model.eval(), r=rank, num_classes=10, 
        #                 increment= self.args['increment'], filepath=self.args['filepath'])

        model.out_dim = 768
        return model

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )

            self._init_train(train_loader, test_loader, optimizer, scheduler)

        else:
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            self._network.backbone = self.update_network(index=False)  # !!!!!!!
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)       
            self._network.to(self._device) 

            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate_per_task_not_include_init"][self._cur_task-1],
                momentum=0.9,
            )  # 1e-5
            print("lr_current_task_init: ", self.args["lrate_per_task_not_include_init"][self._cur_task-1])
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        try:

            tmp_loader = DataLoader(
                self.train_loader.dataset,
                batch_size=self.args["batch_size"],
                shuffle=False, num_workers=0, drop_last=False
            )

            cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            np_state, py_state = np.random.get_state(), random.getstate()

            sharp = self._estimate_task_sharpness_sam(self._network, tmp_loader, rho=0.05, max_batches=10)
            logging.info(f"[Sharpness] Task {self._cur_task}: {sharp:.6f}")
            self._save_task_sharpness(sharp)


            torch.set_rng_state(cpu_state); torch.cuda.set_rng_state_all(cuda_state)
            np.random.set_state(np_state); random.setstate(py_state)

            self._network.train()
            for p in self._network.parameters():
                p.grad = None
        except Exception as e:
            logging.warning(f"[Sharpness] failed to compute: {e}")

        save_lora_name = self.args['filepath']

        if len(self._multiple_gpus) > 1:
            self._network.module.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.module.save_fc(save_lora_name, self._cur_task)
        else:
            self._network.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            print("11111111111111111111111111111111111111111111111111")
            print(save_lora_name,"22222222222222222", self._cur_task)
            self._network.save_fc(save_lora_name, self._cur_task)
            self._network.backbone.save_wrap_param(save_lora_name)

    def get_optimizer(self):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                # lr=self.init_lr, 
                self.args["lrate"],
                # weight_decay=self.weight_decay
                betas=(0.9, 0.999)
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.args['min_lr'])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler



    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                ce = F.cross_entropy(logits, targets)

                loss = ce

                # print('@@@@@@@@@@@@@@loss1', loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )
            logging.info(info)
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        custom_epoch = self.args["epochs_per_task_not_include_init"][self._cur_task-1]
        # print(custom_epoch)
        prog_bar = tqdm(range(custom_epoch))
        # prog_bar = tqdm(range(self.args["epochs"]))
        if self._cur_task == 3 :
            a_t = 10
        else:
            a_t = 10
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # logits = self._network(inputs)["logits"]
                logits, ortho_loss = self._network(inputs, ortho_loss=True)
                logits = logits['logits'] 
                # print(counts)
                # print(logits.shape)
                # print(self._known_classes)
                # print(logits[:, self._known_classes :].shape)
                # print(targets)
                fake_targets = targets - self._known_classes
                # print(fake_targets)
                ce = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )

                loss = ce + a_t * torch.mean(ortho_loss).to(self._device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            # if epoch % 5 == 0:
            if True:  
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    custom_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            logging.info(info)
            prog_bar.set_description(info)
        logging.info(info)
