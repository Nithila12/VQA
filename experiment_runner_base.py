from torch.utils.data import DataLoader
import torch
from student_code.coatt_utils import *
from tensorboardX import SummaryWriter
import datetime


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, modeltype='simple'):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250  # Steps

        self._train_dataset_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()
        self.softmax = torch.nn.Softmax(dim=1)
        self.modeltype = modeltype
        self.till = 250
        self.tbname = './RUNS/SBLINE/'+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if self.modeltype == 'coattention':
            self.W2ID, self.EMB = GetEmbeddings()
            self.AnsWords = getAnsWords(
                path='./student_code/supportfiles/CoAttAns.d')
            self.AW2ID = {w: i for i, w in enumerate(self.AnsWords)}
            self.AW2ID['<unk>'] = 1000
            self.till = 1250
            self.tbname = './RUNS/COATT/'+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.writer = SummaryWriter(log_dir=self.tbname)
        if self._cuda:
            print("CUDA Fied")
            self._model = self._model.cuda()

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self, till=1250):
        Acc = []
        print("Validating...")
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            img = batch_data['image']
            qENC = batch_data['question']  # _nBOW']#
            ground_truth_answer = batch_data['answer']
            if self.modeltype == 'coattention':
                qENC = Sentence2Embeddings(qENC, self.W2ID, self.EMB)
                ground_truth_answer = Answer2OneHot(
                    ground_truth_answer, self.AW2ID)
            predicted_ans = self._model(img.cuda(), qENC.cuda())
            predicted_answer = self.softmax(predicted_ans)
            tpk, tpkvals = torch.topk(predicted_answer, k=1, dim=1)
            gttpk, gtpkvals = torch.topk(
                ground_truth_answer.cuda(), k=1, dim=1)
            # print(tpkvals,gtpkvals)
            acc = (tpkvals == gtpkvals).double().mean()
            Acc.append(acc.item())
            if batch_id >= till:
                print("Done")
                return sum(Acc)/len(Acc)
        return sum(Acc)/len(Acc)
        raise NotImplementedError()

    def train(self):
        # print("Hello")
        # for pg in self.optim.param_groups:
        #     print("Learning rate is ",pg['lr'])
        #     pg['lr']*=0.5
        #     print("Learning rate is ",pg['lr'])
        # return None
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # This block runs the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                img = batch_data['image']
                qENC = batch_data['question']  # _nBOW']
                ground_truth_answer = batch_data['answer']
                if self.modeltype == 'coattention':
                    qENC = Sentence2Embeddings(qENC, self.W2ID, self.EMB)
                    ground_truth_answer = Answer2OneHot(
                        ground_truth_answer, self.AW2ID)
                predicted_answer = self._model(img.cuda(), qENC.cuda())
                # ============

                # Optimize the model according to the predictions
                loss = self._optimize(
                    predicted_answer, ground_truth_answer.cuda())

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch,
                          batch_id, num_batches, loss))
                    self.writer.add_scalar('train/loss', loss, current_step)

                if current_step % self._test_freq == 0:
                    # self._model.eval()
                    val_accuracy = self.validate(till=self.till)
                    print("Epoch: {} has val accuracy {}".format(
                        epoch, val_accuracy))
                    self.writer.add_scalar(
                        'Val/accuracy', val_accuracy, current_step)
                    # Changing the learning rate to half for every epoch:
            for pg in self.optim.param_groups:
                pg['lr'] *= 0.5
                try:
                    self.writer.add_scalar(
                        'learning_rate', pg['lr'], current_step)
                except Exception as e:
                    print(e)
            val_accuracy = self.validate()
            print("complete accuracy")
            print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
