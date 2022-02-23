from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
import torch
import numpy as np


class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
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
        self.model = SimpleBaselineNet(
            len(train_D.SetQdict)+1000, len(train_D.SetAdict)).cuda()
        # self.CE = torch.nn.CrossEntropyLoss()
        # self.NLL = torch.nn.NLLLoss()
        self.BCE = torch.nn.BCELoss()
        self.CE = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

        super().__init__(train_dataset, val_dataset, self.model, batch_size,
                         num_epochs, num_data_loader_workers, modeltype='simple')

    def _optimize(self, predicted_answers, true_answer_ids):
        # print(predicted_answers.shape, true_answer_ids.long().shape)
        ll = self.CE(predicted_answers, true_answer_ids.argmax(
            1).long())  # Nx14000 -> Nx1
        # ll = self.BCE(self.softmax(predicted_answers), true_answer_ids)
        self.optim.zero_grad()
        ll.backward()
        self.optim.step()
        return ll

        raise NotImplementedError()
