from __future__ import print_function
import argparse
import time
import os
import datetime
from utils import utils
from utils.utils import OptimWithSheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models.function import HLoss
from models.function import BetaMixture1D
from models.function import CrossEntropyLoss
from models.basenet import *
import copy
from utils.utils import inverseDecayScheduler, CosineScheduler, StepScheduler, ConstantScheduler
from utils.measure import get_measures
import torch.utils.data as data
import torchvision
import tqdm
from torch.utils.data import Dataset
import os, glob
from torchvision.io import read_image, ImageReadMode
from PIL import Image
from utils.util_praph import show_tsne
import numpy as np
# id_dict = {}
# for i, line in enumerate(open('data/tiny-imagenet-200/wnids.txt', 'r')):
#     id_dict[line.replace('\n', '')] = i

class ImageNetCustomDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        if train:
            
            self.data = np.load(os.path.join(root, 'train_images.npy'))
            self.targets = np.load(os.path.join(root, 'train_labels.npy'))
        else:
            self.data = np.load(os.path.join(root, 'test_images.npy'))
            self.targets = np.load(os.path.join(root, 'test_labels.npy'))
            
        self.target_transform = target_transform
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        targets = self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, targets
    
class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("data/tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('data/tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label
    
class ACRE():
    def __init__(self, args, num_class, src_dset, target_dset, test_transforms):
        self.model = 'ACRE'
        self.args = args
        self.all_num_class = num_class
        self.known_num_class = num_class - 1
        self.dataset = args.dataset
        self.src_dset = src_dset
        self.target_dset = target_dset
        self.device = self.args.device
        self.test_transforms = test_transforms
        self.flag_entropy = False

        self.build_model_init()
        self.ent_criterion = HLoss()
        self.bmm_model = self.cont = self.k = 0
        self.bmm_model_maxLoss = torch.log(torch.FloatTensor([self.known_num_class])).to(self.device)
        self.bmm_model_minLoss = torch.FloatTensor([0.0]).to(self.device)
        self.bmm_update_cnt = 0


        self.src_train_loader, self.src_val_loader, self.src_test_loader, self.src_train_idx = src_dset.get_loaders(
            class_balance_train=True)
        self.target_train_loader, self.target_val_loader, self.target_test_loader, self.tgt_train_idx = target_dset.get_loaders()

        self.num_batches = min(len(self.src_train_loader), len(self.target_train_loader))

        self.cutoff = False



    def build_model_init(self):
        self.G, self.E, self.C = utils.get_model_init(self.args, known_num_class=self.known_num_class, all_num_class=self.all_num_class)
        if self.args.cuda:
            self.G.to(self.args.device)
            self.E.to(self.args.device)
            self.C.to(self.args.device)

        scheduler = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                  max_iter=self.args.warmup_iter)

        if 'vgg' == self.args.net:
            for name, param in self.G.named_parameters():
                if 'lower' in name:
                    param.requires_grad = False
                elif 'upper' in name:
                    param.requires_grad = False
            params = list(list(self.G.linear1.parameters()) + list(self.G.linear2.parameters()) + list(
                self.G.bn1.parameters()) + list(self.G.bn2.parameters()))
        else:
            params = list(self.G.parameters())

        self.opt_w_g = OptimWithSheduler(optim.SGD(params, lr=self.args.g_lr * self.args.e_lr, weight_decay=5e-4, momentum=0.9,
                               nesterov=True), scheduler)
        self.opt_w_e = OptimWithSheduler(optim.SGD(self.E.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_w_c = OptimWithSheduler(optim.SGD(self.C.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)


    def build_model(self):
        def weights_init_bias_zero(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)

        _, _, self.E, self.DC = utils.get_model(self.args, known_num_class=self.known_num_class, all_num_class=self.all_num_class, domain_dim=3)

        self.DC.apply(weights_init_bias_zero)

        if self.args.cuda:
            self.E.to(self.args.device)
            self.DC.to(self.args.device)

        SCHEDULER = {'cos': CosineScheduler, 'step': StepScheduler, 'id': inverseDecayScheduler, 'constant':ConstantScheduler}
        scheduler = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.num_batches*self.args.training_iter)
        scheduler_dc = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.num_batches*self.args.training_iter*self.args.update_freq_D)

        if 'vgg' == self.args.net:
            for name,param in self.G.named_parameters():
                if 'lower' in name:
                    param.requires_grad = False
                elif 'upper' in name:
                    param.requires_grad = False
            params = list(list(self.G.linear1.parameters()) + list(self.G.linear2.parameters()) + list(
                self.G.bn1.parameters()) + list(self.G.bn2.parameters()))
        else:
            params = list(self.G.parameters())

        self.opt_g = OptimWithSheduler(
            optim.SGD(params, lr=self.args.g_lr * self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_c = OptimWithSheduler(
            optim.SGD(self.C.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_dc = OptimWithSheduler(
            optim.SGD(self.DC.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler_dc)

        scheduler_e = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                     max_iter=self.num_batches*self.args.training_iter)
        self.opt_e = OptimWithSheduler(
            optim.SGD(self.E.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True),
            scheduler_e)

    def network_initialization(self):
        if 'resnet' in self.args.net:
            try:
                self.E.fc.reset_parameters()
                self.E.bottleneck.reset_parameters()
            except:
                self.E.fc.reset_parameters()
        elif 'vgg' in self.args.net:
            try:
                self.E.fc.reset_parameters()
                self.E.bottleneck.reset_parameters()
            except:
                self.E.fc.reset_parameters()

    def train_init_ours(self):
        print('train_init starts')
        t1 = time.time()
        epoch_cnt =0
        step=0
        while step < self.args.warmup_iter + 1:
            self.G.train()
            self.E.train()
            self.C.train()
            epoch_cnt +=1
            joint_loader = zip(self.src_train_loader, self.target_train_loader)
            for batch_idx, (((img_s, img_s_og, img_s_aug), label_s, _), ((img_t, img_t_og, img_t_aug), label_t, index_t)) in enumerate(joint_loader):
            #for batch_idx, ((_, _, img_s_aug), label_s, _) in enumerate(self.src_train_loader):
                if self.args.cuda:
                    img_s = img_s_aug[0]
                    img_t = img_t_aug[0]
                    img_s = Variable(img_s.to(self.args.device))
                    label_s = Variable(label_s.to(self.args.device))
                    img_t = Variable(img_t.to(self.args.device))
                    label_t = Variable(label_t.to(self.args.device))

                step += 1
                if step % 10000 == 0:
                    tem_duration = str(datetime.timedelta(seconds=time.time() - t1))[:7]
                    print('step: ' + str(step) + '  time: ' + str(tem_duration))
                if step >= self.args.warmup_iter + 1:
                    break

                self.opt_w_g.zero_grad()
                self.opt_w_e.zero_grad()
                self.opt_w_c.zero_grad()
                feat_s = self.G(img_s)
                out_s = self.E(feat_s)
                feat_t = self.G(img_t)
                out_t = self.E(feat_t)

                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / (self.known_num_class)
                loss_s = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_s, dim=1))

                label_t_onehot = nn.functional.one_hot(label_t, num_classes=self.known_num_class)
                label_t_onehot = label_t_onehot * (1 - self.args.ls_eps)
                label_t_onehot = label_t_onehot + self.args.ls_eps / (self.known_num_class)
                loss_t = CrossEntropyLoss(label=label_t_onehot, predict_prob=F.softmax(out_t, dim=1))

                out_Cs = self.C(feat_s)
                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / (self.all_num_class)
                loss_Cs = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_Cs, dim=1))

                loss = loss_s + loss_Cs + loss_t

                loss.backward()
                self.opt_w_g.step()
                self.opt_w_e.step()
                self.opt_w_c.step()
                self.opt_w_g.zero_grad()
                self.opt_w_e.zero_grad()
                self.opt_w_c.zero_grad()

        duration = str(datetime.timedelta(seconds=time.time() - t1))[:7]
        result_dir = 'results/save_init_model/warmup_iter_' + str(self.args.dataset) + "_" + str(self.args.warmup_iter)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        torch.save(self.G.state_dict(), f'{result_dir}/init_G.pth')
        torch.save(self.E.state_dict(), f'{result_dir}/init_E.pth')
        torch.save(self.C.state_dict(), f'{result_dir}/init_C.pth')
        print('train_init end with duration: %s'%duration)



    def train_load_model(self):
        print('Load init model')
        result_dir = 'results/save_init_model/warmup_iter_' + str(self.args.dataset) + "_" + str(self.args.warmup_iter) + "_best"
        # result_dir = 'results/save_init_model/warmup_iter_' + str(self.args.warmup_iter) + "_best"
        self.G.load_state_dict(torch.load(f'{result_dir}/init_G.pth'))
        self.E.load_state_dict(torch.load(f'{result_dir}/init_E.pth'))
        self.C.load_state_dict(torch.load(f'{result_dir}/init_C.pth'))





    def train(self):
        print('Train Starts')
        t1 = time.time()
        for epoch in range(1, self.args.training_iter):
            joint_loader = zip(self.src_train_loader, self.target_train_loader)
            alpha = float((float(2) / (1 + np.exp(-10 * float((float(epoch) / float(self.args.training_iter)))))) - 1)
            for batch_idx, (((img_s, img_s_og, _), label_s, _), ((img_t, img_t_og, img_t_aug), label_t, index_t)) in enumerate(joint_loader):
                #print(batch_idx)
                self.G.train()
                self.C.train()
                self.DC.train()
                self.E.train()
                if self.args.cuda:
                    img_s = Variable(img_s.to(self.args.device))
                    label_s = Variable(label_s.to(self.args.device))
                    img_t = Variable(img_t.to(self.args.device))
                    img_t_og = Variable(img_t_og.to(self.args.device))
                    img_t_aug = Variable(img_t_aug[0].to(self.args.device))
                    label_t = Variable(label_t.to(self.args.device))

                self.freeze_GC()
                
                out_t_free = self.C_freezed(self.G_freezed2(img_t_og)).detach()
                out_t_free_2 = self.E_freezed(self.G_freezed(img_t_og)).detach()
                w_unk_posterior = self.compute_probabilities_batch(out_t_free, out_t_free_2, self.args.lm)

                w_k_posterior = 1 - w_unk_posterior
                w_k_posterior = w_k_posterior.to(self.args.device)
                w_unk_posterior = w_unk_posterior.to(self.args.device)

                #########################################################################################################
                for d_step in range(self.args.update_freq_D):
                    self.opt_dc.zero_grad()
                    feat_s = self.G(img_s).detach()
                    out_ds = self.DC(feat_s)
                    label_ds = Variable(torch.zeros(img_s.size()[0], dtype=torch.long).to(self.args.device))
                    label_ds = nn.functional.one_hot(label_ds, num_classes=3)
                    loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))  # self.criterion(out_ds, label_ds)

                    label_dt_known = Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                    label_dt_known = nn.functional.one_hot(label_dt_known, num_classes=3)
                    label_dt_unknown = 2 * Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                    label_dt_unknown = nn.functional.one_hot(label_dt_unknown, num_classes=3)
                    feat_t = self.G(img_t).detach()
                    out_dt = self.DC(feat_t)
                    label_dt = w_k_posterior[:, None] * label_dt_known + w_unk_posterior[:, None] * label_dt_unknown
                    loss_dt = CrossEntropyLoss(label=label_dt, predict_prob=F.softmax(out_dt, dim=1))
                    loss_D = self.args.lambda_G * (loss_ds + loss_dt)
                    loss_D.backward()
                    if self.args.opt_clip >0.0:
                        torch.nn.utils.clip_grad_norm_(self.DC.parameters(), self.args.opt_clip)
                    self.opt_dc.step()
                    self.opt_dc.zero_grad()
                #########################################################################################################
                for _ in range(self.args.update_freq_G):
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()
                    feat_s = self.G(img_s)
                    out_ds = self.DC(feat_s)
                    loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))
                    feat_t = self.G(img_t)
                    out_dt = self.DC(feat_t)
                    label_dt = w_k_posterior[:, None] * label_dt_known - w_unk_posterior[:, None] * label_dt_unknown
                    loss_dt = CrossEntropyLoss(label=label_dt, predict_prob=F.softmax(out_dt, dim=1))
                    loss_G = - loss_ds - loss_dt
                    #########################################################################################################
                    out_Cs = self.C(feat_s)
                    label_Cs_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                    label_Cs_onehot = label_Cs_onehot * (1 - self.args.ls_eps)
                    label_Cs_onehot = label_Cs_onehot + self.args.ls_eps / (self.all_num_class)
                    loss_cls_Cs = CrossEntropyLoss(label=label_Cs_onehot, predict_prob=F.softmax(out_Cs, dim=1))

                    label_unknown = (self.known_num_class) * Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                    label_unknown = nn.functional.one_hot(label_unknown, num_classes=self.all_num_class)
                    label_unknown_lsr = label_unknown * (1 - self.args.ls_eps)
                    label_unknown_lsr = label_unknown_lsr + self.args.ls_eps / (self.all_num_class)

                    feat_t_aug = self.G(img_t_aug)
                    out_Ct_aug = self.C(feat_t_aug)
                    if self.cutoff:
                        w_unk_posterior[w_unk_posterior < self.args.threshold] = 0.0
                        w_k_posterior[w_k_posterior < self.args.threshold] = 0.0

                    loss_cls_Ctu = alpha*CrossEntropyLoss(label=label_unknown_lsr, predict_prob=F.softmax(out_Ct_aug, dim=1),
                                                    instance_level_weight=w_unk_posterior)


                    targets_u_onehot = nn.functional.one_hot(label_t, num_classes=self.all_num_class)

                    mask2 = w_k_posterior.ge(self.args.threshold).float()

                    loss_ent_Ctk = CrossEntropyLoss(label=targets_u_onehot,
                                                    predict_prob=F.softmax(out_Ct_aug, dim=1),
                                                    instance_level_weight=mask2)
                    loss = loss_cls_Cs + self.args.lambda_G * loss_G + self.args.lambda_ent_Ctk * loss_ent_Ctk + self.args.lambda_cls_Ctu * loss_cls_Ctu

                    loss.backward()
                    self.opt_g.step()
                    self.opt_c.step()
                    self.opt_e.step()
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()

            if (epoch % 2 == 0):
                auroc, aupr, fprs, detection = self.OODtest(score=self.args.OOD_Score, epoch=epoch)
                print("auroc:{}, aupr:{}, fprs:{}, detection:{}".format(auroc, aupr, fprs, detection))


    def compute_probabilities_batch(self, out_t, out_t2, lm):
        m = torch.nn.Softmax(dim=-1).cuda()
        msp1 = m(out_t)[:, -1]
        msp2, _ = torch.max(m(out_t2), dim=-1)
        msp2 = 1-msp2
        msp = lm * msp1 + (1-lm)*msp2
        msp = (msp - torch.min(msp)) / (torch.max(msp) - torch.min(msp))
        return msp


    def freeze_GE(self):
        self.G_freezed = copy.deepcopy(self.G)
        self.E_freezed = copy.deepcopy(self.E)

    def freeze_GC(self):
        self.G_freezed2 = copy.deepcopy(self.G)
        self.C_freezed = copy.deepcopy(self.C)



    def OODtest(self, score="K+1", epoch=0):
        IDset = torchvision.datasets.CIFAR10('./data', train=False, transform=self.test_transforms, download=True)

        OODset = torchvision.datasets.CIFAR100('./data', train=False, transform=self.test_transforms, download=True)
        OODset.data = np.load("./data/places365_test.npy").transpose((0, 2, 3, 1))
        openset_noise_label = np.random.choice(list(range(10)), size=len(OODset.data), replace=True)
        OODset.targets = openset_noise_label

        IDloader = data.DataLoader(IDset, batch_size=128, shuffle=False, num_workers=0,
                                     drop_last=False)
        OODloader = data.DataLoader(OODset, batch_size=128, shuffle=False, num_workers=0,
                                    drop_last=False)
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        _score_in = []
        _score_out = []
        _right_score = []
        _wrong_score = []
        self.G.eval()
        self.C.eval()
        self.E.eval()

        if score=='K+1':
            features_arr_in, features_arr_out, labels_in = [], [], []
            with torch.no_grad():
                for batch_idx, (inputs, label) in enumerate(IDloader):
                    inputs = inputs.cuda()
                    output_in = self.G(inputs)
                    features_arr_in.append(output_in.detach().cpu())
                    labels_in.append(label.detach().cpu())
                    output_in = self.C(output_in)
                    smax_in = to_np(F.softmax(output_in, dim=1))
                    _score_in.append(-smax_in[:, -1])  # n_class=11
                features_arr_in = torch.cat(features_arr_in, 0)
                labels_in = torch.cat(labels_in, 0)
            with torch.no_grad():
                for batch_idx, (inputs, _) in enumerate(OODloader):
                    inputs = inputs.cuda()
                    output_out = self.G(inputs)
                    features_arr_out.append(output_out.detach().cpu())
                    output_out = self.C(output_out)
                    smax_out = to_np(F.softmax(output_out, dim=1))
                    _score_out.append(-smax_out[:, -1])  # n_class=11
                features_arr_out = torch.cat(features_arr_out, 0)
                features_arr_out = features_arr_out[:4000]
            score_in = concat(_score_in).copy()
            score_out = concat(_score_out).copy()
            save_path = './cifar10_places365_' + str(epoch) + '_.pdf'
            show_tsne(features_arr_in, features_arr_out, save_path, labels_in)

        measures = get_measures(score_in, score_out)
        return measures[0], measures[1], measures[2], measures[3]

    def OODtest_othersets(self, score="K+1"):
        test_ood_sets = ['SVHN', 'LSUN-C', 'LSUN-R', 'Texture', 'Places365', 'iSUN']
        for i in test_ood_sets:
            IDset = torchvision.datasets.CIFAR10('./data', train=False, transform=self.test_transforms, download=True)
            OODset = torchvision.datasets.CIFAR100('./data', train=False, transform=self.test_transforms, download=True)
            test_ood_set = i
            if test_ood_set == 'iSUN':
                OODset = torchvision.datasets.ImageFolder(root="./other_datasets/iSUN",
                                            transform=self.test_transforms)

            if test_ood_set == 'Places365':
                OODset = torchvision.datasets.ImageFolder(root="./other_datasets/places365/",
                                            transform=self.test_transforms)

            if test_ood_set == 'Texture':
                OODset = torchvision.datasets.ImageFolder(root="./other_datasets/dtd/images",
                                            transform=self.test_transforms)

            if test_ood_set == 'SVHN':
                OODset = torchvision.datasets.SVHN('./other_datasets/svhn', split="test", transform=self.test_transforms, download=True)

            if test_ood_set == 'LSUN-C':
                OODset = torchvision.datasets.ImageFolder(root="./other_datasets/LSUN_C",
                                            transform=self.test_transforms)

            if test_ood_set == 'LSUN-R':
                OODset = torchvision.datasets.ImageFolder(root="./other_datasets/LSUN_resize",
                                            transform=self.test_transforms)


            IDloader = data.DataLoader(IDset, batch_size=128, shuffle=False, num_workers=0,
                                       drop_last=False)
            OODloader = data.DataLoader(OODset, batch_size=128, shuffle=False, num_workers=0,
                                        drop_last=False)
            to_np = lambda x: x.data.cpu().numpy()
            concat = lambda x: np.concatenate(x, axis=0)

            _score_in = []
            _score_out = []
            _right_score = []
            _wrong_score = []
            self.G.eval()
            self.C.eval()
            self.E.eval()

            if score == 'K+1':
                with torch.no_grad():
                    for batch_idx, (inputs, _) in enumerate(IDloader):
                        inputs = inputs.cuda()
                        output_in = self.G(inputs)
                        output_in = self.C(output_in)
                        smax_in = to_np(F.softmax(output_in, dim=1))
                        _score_in.append(-smax_in[:, -1])  # n_class=11
                with torch.no_grad():
                    for batch_idx, (inputs, _) in enumerate(OODloader):
                        inputs = inputs.cuda()
                        output_out = self.G(inputs)
                        output_out = self.C(output_out)
                        smax_out = to_np(F.softmax(output_out, dim=1))
                        _score_out.append(-smax_out[:, -1])  # n_class=11
                score_in = concat(_score_in).copy()
                score_out = concat(_score_out).copy()
            measures = get_measures(score_in, score_out)
            print("ood set:", test_ood_set, measures[0], measures[1], measures[2], measures[3], )
        return measures[0], measures[1], measures[2], measures[3]


