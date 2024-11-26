import torch
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Test:
    def __init__(self, configs):


        self.net = configs['netname']('_')
        # self.net = configs['netname']()

        self.test = configs['dataset']['test']
        self.val_dataloader = torch.utils.data.DataLoader(self.test,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=8)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pth = configs['pth_repo']
        self.sava_path = configs['test_path']
        self.print_staistaic_text = self.sava_path + 'print_staistaic_text.txt'

    def start(self):
        print("Loading .......   path:{}".format(self.pth))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.load(self.pth, map_location=device)

        # remove `module.` if exist
        new_state_dict = {}
        for key, value in state['model'].items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value


        self.net.load_state_dict(new_state_dict)
        self.net.to(self.device)


        accuracy = self.val_step(self.pth[-5],self.val_dataloader)

        return accuracy

    def val_step(self, epoch, dataset):
        print('-----------------start test--------------------')


        self.csv = []

        self.net = self.net.eval()
        star_time = time.time()

        for i, data in enumerate(dataset):
            images = data[0].to(self.device)
            labels = data[1].to(self.device)
            with torch.no_grad():
                prediction = self.net(images)
                p1 = torch.argmax(prediction, dim=1).to(self.device)  # operate on dim
                l1 = labels.to(self.device)


                temp_onlylable = torch.cat([l1, p1], dim=-1)
                self.csv.append(temp_onlylable.cpu().detach().numpy().squeeze())


        duration = time.time() - star_time
        speed = 1 / (duration / len(dataset))
        print('avg_time:%.5f, sample_per_s:%.5f' %
                        (duration / len(dataset), speed))

        file_handle = open(self.print_staistaic_text, mode='a')

        file_handle.write('-----------------start test--------------------')
        file_handle.write('\n')
        file_handle.write('avg_time:%.5f, sample_per_s:%.5f' %
                        (duration / len(dataset), speed))

        file_handle.write('\n')
        file_handle.close()

        self.net = self.net.train()
        accuracy = self.tocsv(epoch)

        print('-----------------test over--------------------')

        return accuracy


    def tocsv(self, epoch):

        np_data = np.array(self.csv)

        label = np_data[:,:1]
        pred = np_data[:,1:]


        # Cal OA,F1
        precision = precision_score(label, pred, average='macro')
        recall = recall_score(label, pred, average='macro')
        f1 = f1_score(label, pred, average='macro')
        accuracy = accuracy_score(label, pred)

        #
        print(
            'epoch:{} test_accuracy:{}'.format(epoch,
                        accuracy)
        )
        print(
            'epoch:{} test_precision:{}'.format(epoch,
                        precision)
        )
        print(
            'epoch:{} test_recall:{}'.format(epoch,
                        recall)
        )
        print(
            'epoch:{} test_f1:{}'.format(epoch,
                        f1)
        )


        file_handle = open(self.print_staistaic_text, mode='a')


        file_handle.write('epoch:{}test_accuracy:{}'.format(epoch,
                                                    accuracy ))
        file_handle.write('\n')

        file_handle.write('epoch:{}test_precision:{}'.format(epoch,
                                                    precision ))
        file_handle.write('\n')

        file_handle.write('epoch:{}test_recall:{}'.format(epoch,
                                                    recall))
        file_handle.write('\n')

        file_handle.write('epoch:{}test_f1:{}'.format(epoch,f1))
        file_handle.write('\n')

        file_handle.write('-----------------test over--------------------')
        file_handle.write('\n')

        file_handle.close()

        np.savetxt(self.sava_path + str(epoch)+'_pred.csv', np_data, delimiter=',')

        return accuracy
