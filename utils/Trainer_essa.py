import copy

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
import logging
import time

from utils.model import prepare_model
from utils.data import datatypes
from utils.data import loadertypes
from utils.model_para import filter_para
from utils.loss import MyLosses
from utils.evaluate import Evaluator
from utils.lr import LR_Scheduler
from torch.utils.data import DataLoader, ConcatDataset
from utils.data.my_dataset import SingleClass
from utils.model.examplar import ExemplarDataset
from utils.model import backbones
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms




class Trainer(object):
    def __init__(self, args):
        self.args = args
        prepare_data = datatypes[args.dataset_type]
        self.dataset = prepare_data(args.dataset_name, args)
        self.dataloader = loadertypes[args.loader_type]
        self.label_per_task = [list(np.array(range(args.base_class)))] + [list(np.array(range(args.way)) +
                                                                               args.way * task_id + args.base_class)
                                                                          for task_id in range(args.tasks)]

        # Define model and optimizer
        model = prepare_model(args)
        optim_para = filter_para(model, args, args.lr)
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(optim_para, weight_decay=args.weight_decay, nesterov=args.nesterov)
        else:
            optimizer = torch.optim.Adam(optim_para, weight_decay=args.weight_decay)

        self.criterion = MyLosses(weight=None, args=self.args).build_loss(mode=args.loss_type)

        # Define Evaluator
        self.evaluator = Evaluator(args.all_class)

        # Using cuda
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        if torch.cuda.is_available():
            model = model.cuda()
        self.model, self.optimizer = model, optimizer
        self.old_model = None

        # Resuming checkpoint
        self.best_pred = 0.0
        self.test_out = None
        self.ii = 0

        # history of prediction
        self.acc_history = []
        self.forget_history = []
        
        self.protocovs = []
        self.vectors_train = []
        self.labels_train = []

    def training(self, session):
        session_class_last = 0 if session == 0 else (self.args.base_class + self.args.way * (session - 1))
        session_class = self.args.base_class + self.args.way * session
        classes = [session_class_last, session_class]
        train_dataset = self.dataset[0]
        train_dataset.getTrainData(classes)
        train_loader = self.dataloader(train_dataset, self.args)
        epochs = self.args.base_epochs if session == 0 else self.args.new_epochs
        lr = self.args.lr if session == 0 else self.args.lr_new

        if session == 0 and self.args.val:
            self.model.eval()
            para = torch.load(self.args.model_path)
            para_dict = para['state_dict']
            para_dict_re = self.structure_reorganization(para_dict)
            model_dict = self.model.state_dict()
            state_dict = {k: v for k, v in para_dict_re.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
        else:
            if session > 0:
                self.model.eval()
                backbone = backbones[self.args.backbone_name](mode='parallel_adapters').cuda()
                model_dict = backbone.state_dict()
                para_dict = self.model.backbone.state_dict()
                state_dict = {k: v for k, v in para_dict.items() if k in model_dict.keys()}
                model_dict.update(state_dict)
                backbone.load_state_dict(model_dict)
                self.model.backbone = backbone
                self.model.fix_backbone_adapter()

            self.model.classifier.Incremental_learning(session_class)
            self.model = self.model.cuda()
            self.model.train()

            optim_para = filter_para(self.model, self.args, lr)
            if self.args.optim == 'sgd':
                self.optimizer = torch.optim.SGD(optim_para, weight_decay=self.args.weight_decay, nesterov=self.args.nesterov)
            else:
                self.optimizer = torch.optim.Adam(optim_para, weight_decay=self.args.weight_decay)
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.2)

            print('****** step1: Train current task ******')
            
            for epoch in range(epochs):
                
                # train current task
                tbar = tqdm(train_loader)
                train_loss = 0.0
                               
                for i, sample in enumerate(tbar):
                    query_image, query_target = sample[1], sample[2]
                    if self.args.cuda:
                        query_image, query_target = query_image.cuda(), query_target.cuda()

                    self.optimizer.zero_grad()
                    loss = self._compute_loss(query_image, query_target, session_class_last)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    tbar.set_description('Epoch: %d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))
                if not self.args.no_lr_scheduler:
                    self.scheduler.step()

                if (epoch+1) % 10 == 0:
                    accuracy = self.validation(session)
                    print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
                    self.model.train()    
                    
                # train SFAN and update memory
                if (session > 0) and ((epoch+1) % self.args.new_epochs == 0):     
                    print('****** step2: Train SFAN and update memory ******')                   
                    for p in self.model.backbone.parameters():
                        p.requires_grad = False
                    for p in self.model.classifier.parameters():
                        p.requires_grad = False
                    for p in self.model.sfan.parameters():
                        p.requires_grad = True 
                        
                    self.model.sfan.train()

                    # Adam
                    self.sfan_optimizer = torch.optim.Adam(self.model.sfan.parameters(), weight_decay=5e-4, lr=0.001)
                    self.sfan_scheduler = lr_scheduler.StepLR(self.sfan_optimizer, step_size=45, gamma=0.1)
                    
                    for sfan_epoch in range(self.args.new_epochs):                 
                        sfantbar = tqdm(train_loader)
                        sfan_loss = 0.0
                        for j, samp in enumerate(sfantbar):
                            q_image, q_target = samp[1], samp[2]
                            if self.args.cuda:
                                q_image, q_target = q_image.cuda(), q_target.cuda()
                            
                            with torch.no_grad():
                                v = self.model.feature_extractor(q_image)
                                v_old = self.old_model.feature_extractor(q_image)
                            
                            self.sfan_optimizer.zero_grad()
                            loss_sfan = self._compute_sfanloss(v_old, v, q_target)
                            loss_sfan.backward()
                            self.sfan_optimizer.step()
                            sfan_loss += loss_sfan.item()
                            sfantbar.set_description('Epoch: %d, Sfan loss: %.3f' % (sfan_epoch, sfan_loss / (j + 1)))
                            
                        if not self.args.no_lr_scheduler:
                            self.sfan_scheduler.step()
                    
                    # update memory
                    self.vectors_train = torch.from_numpy(np.array(self.vectors_train)).to(torch.float32).cuda()         
                    with torch.no_grad():
                        self.vectors_train = self.model.sfan_extractor(self.vectors_train)
                        
                    pd = self.vectors_train.cpu().numpy()
                    self.vectors_train = pd.tolist()
                    self.vectors_train = [np.array(element) for element in self.vectors_train]
                            
                    prototype = []
                    protocovs = []
                    class_label = []
                    for item in self.class_label:
                        index = np.where(item == self.labels_train)[0]
                        class_label.append(item)
                        feature_classwise = np.array(self.vectors_train)[index, :]
                        prototype.append(np.mean(feature_classwise, axis=0))
                        protocovs.append(np.cov(feature_classwise.T))
                        
                    self.prototype = prototype
                    self.protocovs = protocovs
                    
                    self._build_feature(session-1)
                    
                    for p in self.model.backbone.parameters():
                        p.requires_grad = True
                    for p in self.model.classifier.parameters():
                        p.requires_grad = True
                    for p in self.model.sfan.parameters():
                        p.requires_grad = False    
                    
                    self.model.train()
                    
                    # finetune classifer
                    self._build_fset(train_loader)
                    fset_loader = DataLoader(self._fset, batch_size=self.args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
                    print('****** step3: Finetune classifer ******')                   
                    for p in self.model.backbone.parameters():
                        p.requires_grad = False
                    for p in self.model.classifier.parameters():
                        p.requires_grad = True
                    for p in self.model.sfan.parameters():
                        p.requires_grad = False  
                        
                    self.model.classifier.train()
                    
                    self.fc_optimizer = torch.optim.Adam(self.model.classifier.parameters(), weight_decay=self.args.weight_decay, lr=0.0002)
                    self.fc_scheduler = lr_scheduler.StepLR(self.fc_optimizer, step_size=45, gamma=0.1)

                    self._train_fc(fset_loader, self.fc_optimizer, self.fc_scheduler)

                    for p in self.model.backbone.parameters():
                        p.requires_grad = True
                    for p in self.model.classifier.parameters():
                        p.requires_grad = True
                    for p in self.model.sfan.parameters():
                        p.requires_grad = False   
                    
                    self.model.train()           
                
        self.StatisticSave(self.model, train_loader, session)
        self.afterTrain(session)
            
    def cossim(self, x1, x2):
        cos_sim = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return cos_sim

    def validation(self, session, print=False):
        self.model.eval()
        self.evaluator.reset()
        forget_history = []

        session_class = self.args.base_class + self.args.way * session
        classes = [0, session_class]

        test_dataset = self.dataset[1]
        test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        for i, sample in enumerate(test_loader):
            image, target = sample[1], sample[2]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Test
        confusion_matrix, Acc_class, Acc = self.evaluator.Acc(session_class)
        logging.info('Validation:')
        logging.info('[Session: %d, numImages: %5d]' % (session, i * self.args.batch_size + image.data.shape[0]))
        logging.info("Acc_class:{}, Acc:{} \n".format(Acc_class, Acc))
        if print:
            self.acc_history.append(Acc)
            for j in range(session+1):
                if j == 0:
                    forget_history.append(np.around(np.mean(Acc_class[:self.args.base_class]), decimals=4))
                else:
                    forget_history.append(np.around(np.mean(Acc_class[self.args.base_class+self.args.way*(j-1): self.args.base_class+self.args.way*j]), decimals=4))
            self.forget_history.append(forget_history)
            if session == (self.args.session-1):
                forget_avg = 0
                for ff in range(self.args.session-1):
                    forget_avg = forget_avg + self.forget_history[ff][ff] - self.forget_history[self.args.session-1][ff]
                forget_avg /= (self.args.session-1)
                logging.info("Avg_for:{} \n".format(forget_avg))
                logging.info("Avg_acc:{} \n".format(np.mean(self.acc_history)))
                logging.info("Acc:{} \n".format(list(self.acc_history)))

        return Acc

    def afterTrain(self, current_task):
        torch.save({'state_dict': self.model.state_dict()},
                   os.path.join(self.args.dir_name,
                                'model_' + self.args.dataset_name + '_' + str(self.args.way) + '_' + str(
                                    current_task) + '.pth'))
                                  
        self.old_model = copy.deepcopy(self.model)
        self.old_model.cuda()
        self.old_model.eval()

        if current_task > 0:
            model_dict = self.model.state_dict()
            for k, v in model_dict.items():
                if 'adapter' in k:
                    k_conv3 = k.replace('adapter', 'conv')
                    model_dict[k_conv3] = model_dict[k_conv3] + F.pad(v, [1, 1, 1, 1], 'constant', 0)
                    model_dict[k] = torch.zeros_like(v)
            self.model.load_state_dict(model_dict)
        
        self._build_feature(current_task)
            
    def _compute_loss(self, imgs, target, old_class=0):
        if self.old_model is None:
            output = self.model(imgs)
            loss_cls = nn.CrossEntropyLoss()(output / 0.1, target)
            return loss_cls
        else:
            feature = self.model.feature_extractor(imgs)
            feature_old = self.old_model.feature_extractor(imgs)
            output = self.model(imgs)
            loss_cls = nn.CrossEntropyLoss()(output / 0.1, target)
            loss_kd = torch.mean(torch.norm(feature - feature_old, p=2, dim=1))

            proto_aug = []
            proto_aug_label = []
            index = list(range(len(self.vectors_train)))
            for _ in range(self.args.batch_size):
                np.random.shuffle(index)
                temp = self.vectors_train[index[0]]
                proto_aug.append(temp)
                proto_aug_label.append(self.labels_train[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().cuda()
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).cuda()
            soft_feat_aug = self.model.classifier(proto_aug, 0)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / 0.1, proto_aug_label)
            
            return loss_cls + 1 * loss_protoAug + self.args.lam * loss_kd

    def _compute_sfanloss(self, vo, vn, target):
        v_old = self.model.sfan_extractor(vo)
        with torch.no_grad():
            v_old_target = self.model.classify(v_old)
        loss_sim = torch.mean(torch.norm(v_old-vn, p=2, dim=1), dim=0)
        loss_c = nn.CrossEntropyLoss()(v_old_target / 0.1, target)
        
        # Manifold Consistency
        proto_aug = []
        proto_aug_label = []
        index = list(range(len(self.vectors_train)))
        for _ in range(self.args.batch_size):
            np.random.shuffle(index)
            temp = self.vectors_train[index[0]]
            proto_aug.append(temp)
            proto_aug_label.append(self.labels_train[index[0]])

        proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().cuda()
        proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).cuda()
        proto_aug1 = self.model.sfan_extractor(proto_aug)
        
        proto_aug = torch.cat((proto_aug, vo), dim=0)
        proto_aug1 = torch.cat((proto_aug1, vn), dim=0)
        
        loss_mc = self.manifold_consis(proto_aug1, proto_aug)
        
        return loss_c + self.args.beta * loss_mc + self.args.alp * loss_sim

    def manifold_consis(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)
        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = torch.nn.functional.normalize(G_t)
        G_diff = G_t - G_s
        loss_mc = (G_diff * G_diff).view(-1, 1).sum(0)  / (bsz)                             
        return loss_mc

    def cosine_loss(self, nf, of):
        nf = nf.view(nf.size(0), -1)
        of = of.view(of.size(0), -1)
        nf = F.normalize(nf, dim=0, p=2)
        nf = F.normalize(nf, dim=0, p=2)
        return torch.mean(1.0 - F.cosine_similarity(nf, of)) 
         
    def structure_reorganization(self, para_dict):
        para_dict_re = copy.deepcopy(para_dict)
        for k, v in para_dict.items():
            if 'bn.weight' in k or 'bn1.weight' in k or 'downsample.1.weight' in k:
                if 'bn.weight' in k:
                    k_conv3 = k.replace('bn', 'conv')
                elif 'bn1.weight' in k:
                    k_conv3 = k.replace('bn1', 'conv1')
                elif 'downsample.1.weight' in k:
                    k_conv3 = k.replace('1', '0')
                k_conv3_bias = k_conv3.replace('weight', 'bias')
                k_bn_bias = k.replace('weight', 'bias')
                k_bn_mean = k.replace('weight', 'running_mean')
                k_bn_var = k.replace('weight', 'running_var')

                gamma = para_dict[k]
                beta = para_dict[k_bn_bias]
                running_mean = para_dict[k_bn_mean]
                running_var = para_dict[k_bn_var]
                eps = 1e-5
                std = (running_var + eps).sqrt()
                t = (gamma / std).reshape(-1, 1, 1, 1)
                para_dict_re[k_conv3] *= t
                para_dict_re[k_conv3_bias] = beta - running_mean * gamma / std
        return para_dict_re

    def StatisticSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (_, images, target) in enumerate(loader):
                feature = model.feature_extractor(images.cuda())
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        prototype = []
        protocovs = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            protocovs.append(np.cov(feature_classwise.T))

        if current_task == 0:
            self.prototype = prototype
            self.protocovs = protocovs
            self.class_label = class_label
        else:
            self.prototype = np.concatenate((self.prototype, prototype), axis=0)
            self.protocovs = np.concatenate((self.protocovs, protocovs), axis=0)
            self.class_label = np.concatenate((self.class_label, class_label), axis=0)

    def _build_feature(self, s):
        self.vectors_train = []
        self.labels_train = []
        
        s_class_last = 0 if s == 0 else (self.args.base_class + self.args.way * (s - 1))
        s_class = self.args.base_class + self.args.way * s
        
        for class_idx in range(0, s_class):

            mean = self.prototype[class_idx]
            cov = self.protocovs[class_idx]
            # cifar100 and tiny 500; imgsub 1300
            num_sampled = int(500)
            self.vectors_train.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
            self.labels_train.append([class_idx]*num_sampled)
        
        self.vectors_train = np.concatenate(self.vectors_train)
        self.labels_train = np.concatenate(self.labels_train)
        
    # build feature set for finetune classifer
    def _build_fset(self, loader):
        features = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (_, images, target) in enumerate(loader):
                feature = self.model.feature_extractor(images.cuda())
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))   
        
        features = np.concatenate((self.vectors_train, features), axis=0)
        labels = np.concatenate((self.labels_train, labels))
        self._fset = FeatureDataset(features, labels)

    # finetune classifer
    def _train_fc(self, fcloader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args.new_epochs))                
        for _, epoch in enumerate(prog_bar):
            fc_loss = 0.0
            for i, (_, inputs, targets) in enumerate(fcloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs = inputs.to(torch.float32)
                logits = self.model.classifier(inputs, 0)
                optimizer.zero_grad()
                lossfc = nn.CrossEntropyLoss()(logits / 0.1, targets)
                lossfc.backward()
                optimizer.step()
                fc_loss += lossfc.item()
                prog_bar.set_description('Epoch: %d, Finetune loss: %.3f' % (epoch, fc_loss / (i + 1)))
            if not self.args.no_lr_scheduler:
                scheduler.step()                
            if (epoch+1) % 10 == 0:
                print('fc_epoch:%d, fc_loss:%.3f' % (epoch, fc_loss / (i+1)))

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return idx, feature, label
