import torch
import torch.nn as nn
from external.googlenet import googlenet as G


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """

    def __init__(self, insize, outsize):
        super().__init__()
        self.GN = G.googlenet(pretrained=True)

        self.outlayer = nn.Linear(insize, outsize)  # 9000, outVocab 14000

    def forward(self, image, question_encoding):
        # TODO
        imgcls = self.GN(image)
        if type(imgcls) == tuple:
            imgcls = imgcls[-1]
        # Nx1000 catting Nx8000 => Nx(1000+8000) => Nx9000
        cat = torch.cat([imgcls, question_encoding], dim=1)
        return self.outlayer(cat)
        raise NotImplementedError()
