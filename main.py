from utils import *
import torch
import time
import torch.nn as nn
import os
import pickle
from tqdm import tqdm
class Main():
    def __init__(self, args, Dataloader):
        self.args = args
        self.lr=self.args.learning_rate
        self.dataloader_gt = Dataloader
        self.epoch = 0

    def save_model(self,epoch):
        model_path= self.args.model_filepath + '/' + self.args.train_model + '_' +\
                                   str(epoch) + '.pth'
        checkpoint = {
            'epoch': epoch,
            'net': self.net,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler
        }
        torch.save(checkpoint, model_path,_use_new_zipfile_serialization=False)
        self.args.load_model=epoch
        modifyArgsfile(self.args.config, 'load_model', epoch)


    def load_model(self):
        if self.args.load_model >= 0:
            self.args.model_save_path = self.args.model_filepath + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.pth'
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path,map_location={'cuda:1': 'cuda:'+str(self.args.gpu)})
                model_epoch = checkpoint['epoch']
                self.epoch = int(model_epoch)+1
                self.net = checkpoint['net']
                self.optimizer = checkpoint['optimizer']
                self.scheduler = checkpoint['scheduler']
                print('Loaded checkpoint at epoch', model_epoch)


    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss(reduce=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,\
        T_max = self.args.num_epochs, eta_min=self.args.eta_min)

    def playtest(self):
        print('Testing begin')
        model_filepath = os.path.join(self.args.model_filepath,"best_model.pth")
        if os.path.exists(model_filepath):
            model_state_dict = torch.load(model_filepath,map_location={'cuda:1': 'cuda:'+str(self.args.gpu)})
            self.net = model_state_dict
            self.net.args = self.args
            test_error, test_final_error, first_erro_test, time = self.test_epoch()
            print('test_error: {:.5f} test_final_error: {:.5f} time: {:.5f}'.format(test_error,test_final_error, time))
        else:
            print("No model weight file!")
            return

    def playEntireTrain(self):
        torch.cuda.empty_cache()
        perf_dict = {
            "whole_model[ADE,FDE]": [1e3, 1e3]
        }
        dict_key = "whole_model[ADE,FDE]"
        print(self.args.load_model)
        model = import_class(self.args.model)
        self.net = model(self.args)
        self.set_optimizer()
        if self.args.load_model >= 0:
            self.load_model()
        else:
            self.epoch=0
            if self.args.using_cuda:
                self.net = self.net.cuda()
        print('Training begin')
        epochs_tqdm = tqdm(range(self.epoch,self.args.num_epochs))
        start = time.time()
        for epoch in epochs_tqdm:
            with torch.no_grad():
                print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))
            train_loss = self.train_epoch(epoch)
            val_error, val_final_error, val_erro_first,valtime = self.val_epoch()
            val_res = [val_error,val_final_error]
            with torch.no_grad():
                # print(f'epoch={epoch} | val_loss={val_loss} | time={valtime}')
                val_loss_logfilepath = os.path.join(self.args.model_filepath, 'val_loss_log.txt')
                cur_valloss_content = f'epoch={epoch+1} | valid_error={val_error} | valid_final={val_final_error} | time={valtime}'
                with open(val_loss_logfilepath, 'a') as file:
                    file.write(cur_valloss_content+ '\n')
            self.scheduler.step()
            if epoch%10 == 0:
                self.save_model(epoch+1)
            with torch.no_grad():
                print('----epoch {} \n train_loss={:.5f}, valid_error={:.3f}, valid_final={:.3f}, valid_first={:.3f}'\
                    .format(epoch, train_loss,val_error, val_final_error,val_erro_first))
                if val_res[1] < perf_dict[dict_key][1]:
                    perf_dict[dict_key][0], perf_dict[dict_key][1] = val_res[0], val_res[1]
                    torch.save(self.net, os.path.join(self.args.model_filepath, "best_model.pth"))
                    with open(os.path.join(self.args.model_filepath, "Performances.pkl"), "wb") as f:
                        pickle.dump(perf_dict, f, 4)
                    print("==>best_model Saved")
            epochs_tqdm.update(1)
        end = time.time()
        traintime = end-start
        epochs_tqdm.close()
        # torch.save(self.net, os.path.join(self.args.model_filepath,f"epoch{self.args.num_epochs}_bs{self.args.batch_size}_wholeModel.pth"))
        train_timeContent = f'train_time = {traintime/3600} H'
        print(train_timeContent)
        with open(val_loss_logfilepath, 'a') as file:
            file.write(train_timeContent + '\n')
            file.write(f'graphconv = {self.args.cluster_num}' + '\n')

    def get_inputsfw(self,batch,epoch,setNum, isval):
        # inputs_gt表示每个时间步的所有轨迹序列的坐标
        # batch_split表示不同场景下轨迹序列的索引
        # nei_lists表示不同场景下轨迹序列的每个时间步（前8）下某行人是否依旧是邻居（10m） 1表示是,0表示否
        if isval:
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_val_batch(batch, epoch)
        else:
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_train_batch(batch,epoch,setNum)  # batch_split:[batch_size, 2]
        #将 inputs_gt 列表中的每个数据元素转换成一个 PyTorch 张量，并将这些张量作为元素放入一个新的元组中。
        inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
        if self.args.using_cuda:
            inputs_gt = tuple([i.cuda() for i in inputs_gt])
        batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
        inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split  # [H, N, 2], [H, N, 2], [B, H, N, N], [N, H]
        return inputs_fw

    def train_epoch(self, epoch):
        self.net.train()
        loss_epoch=0
        ballance_array = balance_mapping(self.dataloader_gt.trainbatchnums,self.dataloader_gt.valbatchnums)
        for index, batch in enumerate(ballance_array):
            start = time.time()

            inputs_fw_0 = self.get_inputsfw(batch[0], epoch, 0, False)
            inputs_fw_1 = self.get_inputsfw(batch[1], epoch, None,True)
            self.net.zero_grad()
            total_loss, full_pre_tra = self.net.forward(inputs_fw_0, inputs_fw_1,batch,iftest=False,ifvisualize=False)

            # if align_loss == 0:
            #     continue
            # param_lambda = 0.6
            # param_beta = 0.
            # totalLoss = predict_loss + ballance_param  * align_loss + (1-ballance_param) * variety_loss

            loss_epoch = loss_epoch + total_loss.item()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            end= time.time()
            with torch.no_grad():
                print('train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f}'.\
                format(index+1,len(ballance_array), epoch, total_loss.item(), end - start))
        train_loss_epoch = loss_epoch / len(ballance_array)
        return train_loss_epoch

    def val_epoch(self):
        self.net.eval()
        error_epoch,final_error_epoch, first_erro_epoch = 0,0,0
        error_epoch_list, final_error_epoch_list, first_erro_epoch_list= [], [], []
        error_cnt_epoch, final_error_cnt_epoch, first_erro_cnt_epoch = 1e-5,1e-5,1e-5
        start = time.time()
        for batch in range(self.dataloader_gt.testbatchnums):
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_test_batch(batch, None)  # batch_split:[batch_size, 2]
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])
            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split
            full_pre_tra = self.net.forward(inputs_fw , [], batch,iftest=True)

            for pre_tra in full_pre_tra:
                error, error_cnt, final_error, final_error_cnt, first_erro,first_erro_cnt = \
                L2forTest(pre_tra, batch_norm_gt[1:, :, :2],self.args.obs_length)
                error_epoch_list.append(error)
                final_error_epoch_list.append(final_error)
                first_erro_epoch_list.append(first_erro)

            first_erro_epoch = first_erro_epoch + min(first_erro_epoch_list)
            final_error_epoch = final_error_epoch + min(final_error_epoch_list)
            error_epoch = error_epoch + min(error_epoch_list)
            error_cnt_epoch = error_cnt_epoch + error_cnt
            final_error_cnt_epoch = final_error_cnt_epoch + final_error_cnt
            first_erro_cnt_epoch = first_erro_cnt_epoch + first_erro_cnt
            error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        end = time.time()
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch,first_erro_epoch/ first_erro_cnt_epoch,end-start

    def test_epoch(self):
        self.net.eval()
        error_epoch, final_error_epoch, first_erro_epoch = 0, 0, 0
        error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        error_cnt_epoch, final_error_cnt_epoch, first_erro_cnt_epoch = 1e-5, 1e-5, 1e-5
        start = time.time()
        for batch in range(self.dataloader_gt.testbatchnums):
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_test_batch(batch,
                                                                                  0)  # batch_split:[batch_size, 2]
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])
            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split
            full_pre_tra = self.net.forward(inputs_fw, [], batch, iftest=True, ifvisualize=True)

            for pre_tra in full_pre_tra:
                error, error_cnt, final_error, final_error_cnt, first_erro, first_erro_cnt = \
                    L2forTest(pre_tra, batch_norm_gt[1:, :, :2], self.args.obs_length)
                error_epoch_list.append(error)
                final_error_epoch_list.append(final_error)
                first_erro_epoch_list.append(first_erro)

            first_erro_epoch = first_erro_epoch + min(first_erro_epoch_list)
            final_error_epoch = final_error_epoch + min(final_error_epoch_list)
            error_epoch = error_epoch + min(error_epoch_list)
            error_cnt_epoch = error_cnt_epoch + error_cnt
            final_error_cnt_epoch = final_error_cnt_epoch + final_error_cnt
            first_erro_cnt_epoch = first_erro_cnt_epoch + first_erro_cnt
            error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        end = time.time()
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch,first_erro_epoch/ first_erro_cnt_epoch,end-start

    def getTrainXandY(self, inputs):
        _, batch_norm_gt, _, _, _ = inputs
        train_x = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length - 1, :,
                                                                :]  # [H, N, 2]
        train_x = train_x.permute(1, 2, 0)  # [N, 2, H]
        train_y = batch_norm_gt[self.args.obs_length:, :, :].permute(1, 2, 0)  # [N, 2, H]
        return train_x, train_y
