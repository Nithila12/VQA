from student_code.coattention_net import CoattentionNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.coatt_vqa_dataset import VqaDataset
import numpy as np
import torch


class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """

    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path, test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers):

        train_D = VqaDataset(image_dir=train_image_dir,
                             question_json_file_path=train_question_path,
                             annotation_json_file_path=train_annotation_path,
                             image_filename_pattern="COCO_train2014_{}.jpg")
        val_D = VqaDataset(image_dir=test_image_dir,
                           question_json_file_path=test_question_path,
                           annotation_json_file_path=test_annotation_path,
                           image_filename_pattern="COCO_val2014_{}.jpg")
        mval = len(val_D)
        # inds30 = np.arange(0,int(0.3*mval))
        # inds70 = np.arange(int(0.3*mval),mval)
        TOT = np.random.randint(0, mval, mval)
        inds30, inds70 = TOT[:int(0.3*mval)], TOT[int(0.3*mval):]
        val70 = torch.utils.data.Subset(val_D, inds70)
        val_dataset = torch.utils.data.Subset(val_D, inds30)
        train_dataset = torch.utils.data.ConcatDataset([train_D, val70])
        # self.model = SimpleBaselineNet(len(train_D.SetQdict)+1000, len(train_D.SetAdict)).cuda()
        self._model = CoattentionNet()
        # self.BCE = torch.nn.BCELoss()
        self.CE = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        # self.optim = torch.optim.RMSprop(self._model.parameters(), lr=4e-4, weight_decay=1e-8,momentum=0.99)#
        self.optim = torch.optim.Adam(
            self._model.parameters(), lr=0.001)  # can be changed
        # self._model = CoattentionNet()

        super().__init__(train_dataset, val_dataset, self._model, batch_size, num_epochs,
                         num_data_loader_workers=num_data_loader_workers, modeltype='coattention')

    def _optimize(self, predicted_answers, true_answer_ids):
        # ll = self.BCE(self.softmax(predicted_answers), true_answer_ids)
        ll = self.CE(predicted_answers, true_answer_ids.argmax(1).long())
        self.optim.zero_grad()
        ll.backward()
        self.optim.step()
        return ll
        raise NotImplementedError()
