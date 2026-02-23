from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataloader.data_utils import get_dataloader
from utils import *

import tool
from inc_net.net import NET
class SAFA(nn.Module):
    def __init__(self, args):
        super(SAFA, self).__init__()
        self.args = args
        tool.args = args
        self.session_time = []
        set_save_path(self)
        self.logs = []
        for arg, value in vars(args).items():
            log_and_print(self, f"{arg}: {value}")
        self.model = NET(self.args, mode=self.args.base_mode)
        if len(self.args.gpu) > 1:
            self.model = nn.DataParallel(self.model, self.args.gpu)
            self.model = self.model.cuda()
        else:
            self.model = self.model.to(args.device)
        self.infer_time_sessions=[0.0]*args.sessions
        if self.args.model_dir is not None:
            log_and_print(self, 'Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            log_and_print(self, 'random init params')
            if self.args.start_session > 0:
                log_and_print(self, 'WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

        print_param_size(self.model, self)

    def train(self, ):
        args = self.args
        max_acc_sessions = [0.0] * args.sessions
        unseen_acc_sessions = [0.0] * args.sessions
        seen_acc_sessions = [0.0] * args.sessions
        # init train statistics
        for session in range(args.start_session, args.sessions):
            if self.args.dataset in ['cifar10dvs', 'dvs128gesture', 'n_caltech101']:
                train_set, trainloader, testloader, train_idx, test_idx = get_dataloader(args, session)
            else:
                train_set, trainloader, testloader = get_dataloader(args, session)
            self.model.load_state_dict(self.best_model_dict)
            if session == 0:  # load base class train img label
                if self.args.dataset in ['cifar10dvs', 'dvs128gesture', 'n_caltech101']:
                    log_and_print(self, f'new classes for this session:{np.unique(train_idx)}')
                else:
                    log_and_print(self, f'new classes for this session:{np.unique(train_set.targets)}')
                optimizer, scheduler = get_optimizer_scheduler(args, self.model.parameters())
                # torch.cuda.synchronize()

                for epoch in range(args.epochs_base):
                    epoch_start = time.time()
                    tl, ta = self.base_train(trainloader, optimizer, scheduler, session, epoch)
                    epoch_end = time.time()

                    log_and_print(self, f"Epoch {epoch + 1}/{args.epochs_base} Cost Time: {epoch_end - epoch_start:.2f} s")
                    tl = tl / len(trainloader)
                    ta = ta / len(trainloader)
                    tsl, tsa = self.test(self.model, testloader, epoch, args, session)
                    # save better model
                    if (tsa * 100) >= max_acc_sessions[session]:
                        max_acc_sessions[session] = float('%.3f' % (tsa * 100))
                        max_acc_epoch = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')

                        self.best_model_dict = deepcopy(self.model.state_dict())
                        log_and_print(self, '********A better model is found!!**********')
                        log_and_print(self, 'Saving model to :%s' % save_model_dir)
                    log_and_print(self, 'best epoch {}, best test acc={:.3f}'.format(max_acc_epoch+1, max_acc_sessions[session]))

                    lrc = scheduler.get_last_lr()[0]
                    log_and_print(self, 'epoch:%d/%d, lr:%.4f, training_loss:%.5f, training_acc:%.5f, test_loss:%.5f, test_acc:%.5f' % (epoch+1,args.epochs_base, lrc, tl, ta, tsl, tsa))
                    print_config(args)
                    scheduler.step()
                # Finish base train

                log_and_print(self, '>>> Finish Base Train <<<')
                log_and_print(self, 'Session {}, Test Best Epoch {}, best test Acc {:.4f}'.format(session, max_acc_epoch+1, max_acc_sessions[session]))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    if self.args.dataset in ['cifar10dvs', 'dvs128gesture', 'n_caltech101']:
                        self.model = self.replace_base_fc(train_set, None, self.model, args)
                    else:
                        self.model = self.replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    log_and_print(self, 'Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    # torch.save(dict(params=self.model.state_dict()), best_model_dir)
                    if len(args.gpu) > 1:
                        self.model.module.mode = 'avg_cos'
                    else:
                        self.model.mode = 'avg_cos'
                    tsl, tsa = self.test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= max_acc_sessions[session]:
                        max_acc_sessions[session] = float('%.3f' % (tsa * 100))
                        log_and_print(self, 'The new best test acc of base session={:.3f}'.format(max_acc_sessions[session]))
            # incremental learning sessions
            else:
                log_and_print(self, "training session: [%d]" % session)
                if len(args.gpu) > 1:
                    self.model.module.mode = self.args.new_mode
                else:
                    self.model.mode = self.args.new_mode
                self.model.eval()
                if self.args.dataset not in ['cifar10dvs', 'dvs128gesture', 'n_caltech101']:
                    trainloader.dataset.transform = testloader.dataset.transform
                if args.new_update is not None:
                    if len(args.gpu) > 1:
                        self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                        self.model.module.subspace_projection(args, session)
                    else:
                        if self.args.dataset not in ['cifar10dvs', 'dvs128gesture', 'n_caltech101']:
                            self.model.update_fc(trainloader, np.unique(train_set.targets), session)
                        else:
                            self.model.update_fc(trainloader, np.unique(train_idx), session)
                        torch.cuda.synchronize(args.device)  # 同步GPU
                        start = time.perf_counter()
                        self.model.subspace_projection(args, session)
                        torch.cuda.synchronize(args.device)  # 同步GPU
                        end = time.perf_counter()
                        subspace_time = end - start
                        self.session_time.append(subspace_time)
                        log_and_print(self, f"GPU elapsed time: {subspace_time*1e3:.4f} ms")

                tsl, (seenac, unseenac, avgac) = self.test(self.model, testloader, 0, args, session)
                # update results and save model
                log_and_print(self, f"Seen Accuracy: {seenac * 100:.3f}%")
                log_and_print(self, f"Unseen Accuracy: {unseenac * 100:.3f}%")

                max_acc_sessions[session] = float('%.3f' % (avgac * 100))
                unseen_acc_sessions[session] = float('%.3f' % (unseenac * 100))
                seen_acc_sessions[session] = float('%.3f' % (seenac * 100))
                self.best_model_dict = deepcopy(self.model.state_dict())
                log_and_print(self, f"Session {session} ==> Seen Acc:{seenac * 100:.3f}%"
                          f"Unseen Acc:{unseenac * 100:.3f}% Avg Acc:{max_acc_sessions[session]}")
                log_and_print(self, 'Session {}, test Acc {:.3f}'.format(session, max_acc_sessions[session]))

        log_and_print(self, 'Base Session Best Epoch {}'.format(max_acc_epoch+1))
        log_and_print(self, f"max_acc: {max_acc_sessions}")
        log_and_print(self, f"Seen acc: {seen_acc_sessions}")
        log_and_print(self, f"Unseen acc: {unseen_acc_sessions}")
        log_and_print(self, f"Harmonic mean: {harm_mean(seen_acc_sessions, unseen_acc_sessions)}")
        log_and_print(self, f"Infer time: {self.infer_time_sessions}")
        log_and_print(self, f"Average time: {sum(self.infer_time_sessions) / len(self.infer_time_sessions)}")
        log_and_print(self, f"Save path is {args.save_path}")
        log_and_print(self, f"SG={args.sg}")
        log_and_print(self, f"Average session time: {sum(self.session_time) / len(self.session_time):.3f} ms")
        logger = log_to_file(os.path.join(args.save_path, 'log.txt'))
        for log_entry in self.logs:
            logger.info(log_entry)
        print_config(args, logger=logger)
        print_config(args,logger,is_end=True)


    def base_train(self, trainloader, optimizer, scheduler, session, epoch):
        args = self.args
        tl, ta = 0.0, 0.0
        self.model = self.model.train()
        # standard classification for pretrain
        tqdm_gen = tqdm(trainloader)
        torch.cuda.reset_peak_memory_stats(device=args.device)
        for i, batch in enumerate(tqdm_gen, 1):
            if len(args.gpu) > 1:
                data, train_label = [_.cuda() for _ in batch]
            else:
                data, train_label = [_.to(args.device) for _ in batch]
            logits = self.model(data, session=session)
            logits = logits[:, :, :args.base_class]
            if len(args.gpu) > 1:
                criterion = nn.CrossEntropyLoss().cuda()
            else:
                criterion = nn.CrossEntropyLoss().to(args.device)

            loss = TET_loss(logits, train_label, criterion, args.means, args.lamb)
            logits = logits.mean(1)

            acc = count_acc(logits, train_label)
            total_loss = loss
            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}/{}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch + 1, args.epochs_base, lrc,
                                                                                       total_loss.item(), acc))
            tl += total_loss.item()
            ta += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device=args.device)
        return tl, ta

    def replace_base_fc(self, trainset, transform, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, num_workers=8, pin_memory=True, shuffle=False)
        if self.args.dataset not in ['cifar10dvs', 'dvs128gesture', 'n_caltech101']:
            trainloader.dataset.transform = transform
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                if len(args.gpu) > 1:
                    data, label = [_.cuda() for _ in batch]
                else:
                    data, label = [_.to(args.device) for _ in batch]
                if len(args.gpu) > 1:
                    model.module.mode = 'encoder'
                else:
                    model.mode = 'encoder'
                embedding = model(data)

                embedding = embedding.mean(1)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        proto_list = []
        for class_index in range(args.base_class):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        proto_list = torch.stack(proto_list, dim=0)
        if len(args.gpu) > 1:
            model.module.fc.weight.data[:args.base_class] = proto_list
        else:
            model.fc.weight.data[:args.base_class] = proto_list
        return model

    def test(self, model, testloader, epoch, args, session):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl, va, va5 = 0.0, 0.0, 0.0
        lgt = torch.tensor([])
        lbs = torch.tensor([])
        torch.cuda.reset_peak_memory_stats(device=args.device)
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                if len(args.gpu) > 1:
                    data, test_label = [_.cuda() for _ in batch]
                else:
                    data, test_label = [_.to(args.device) for _ in batch]
                logits = model(data)

                logits = logits.mean(1)
                logits = logits[:, :test_class]
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                top5acc = count_acc_topk(logits, test_label)
                vl += loss.item()
                va += acc
                va5 += top5acc
                lgt = torch.cat([lgt, logits.cpu()])
                lbs = torch.cat([lbs, test_label.cpu()])
            torch.cuda.synchronize(device=args.device)

            vl = vl / len(testloader)
            va = va / len(testloader)
            va5 = va5 / len(testloader)
            log_and_print(self, 'epo {}/{}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch+1, args.epochs_base, vl, va, va5))
            print_config(args)
            lgt = lgt.view(-1, test_class)
            lbs = lbs.view(-1)

            if session > 0:
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
                cm = confmatrix(lgt, lbs, save_model_dir)
                perclassacc = cm.diagonal()
                seenac = np.mean(perclassacc[:args.base_class])
                unseenac = np.mean(perclassacc[args.base_class:])
                log_and_print(self, f"Seen Acc:{seenac}  Unseen Acc:{unseenac}")
                return vl, (seenac, unseenac, va)
            else:
                return vl, va



def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, :], labels)
    Loss_es = Loss_es / T
    if lamb != 0:
        MMDLoss = nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd
