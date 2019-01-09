import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision
from models.fracPickup import fracPickup
import numpy.random as npr

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=128):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size,bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size+num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        self.fracPickup = fracPickup()

    def forward(self, prev_hidden, feats, cur_embeddings, test=False):
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size

        feats_proj = self.i2h(feats.view(-1,nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        emition = self.score(F.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB).transpose(0,1)

        alpha = F.softmax(emition, 1) # nB * nT

        if not test:
            alpha_fp = self.fracPickup(alpha.unsqueeze(1).unsqueeze(2)).squeeze()
            context = (feats * alpha_fp.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0) # nB * nC
            if len(context.size()) == 1:
                context = context.unsqueeze(0)
            context = torch.cat([context, cur_embeddings], 1)
            cur_hidden = self.rnn(context, prev_hidden)
            return cur_hidden, alpha_fp
        else:
            context = (feats * alpha.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0) # nB * nC
            if len(context.size()) == 1:
                context = context.unsqueeze(0)
            context = torch.cat([context, cur_embeddings], 1)
            cur_hidden = self.rnn(context, prev_hidden)
            return cur_hidden, alpha

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_embeddings=128):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_embeddings)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes)
        self.char_embeddings = Parameter(torch.randn(num_classes+1, num_embeddings))
        self.num_embeddings = num_embeddings
        self.processed_batches = 0
        self.num_classes = num_classes

    # targets is nT * nB
    def forward(self, feats, text_length, text, test=False):
        if not test:
            self.processed_batches = self.processed_batches + 1
            nT = feats.size(0)
            nB = feats.size(1)
            nC = feats.size(2)
            hidden_size = self.hidden_size
            input_size = self.input_size
            assert(input_size == nC)
            assert(nB == text_length.numel())

            num_steps = text_length.data.max()
            num_labels = text_length.data.sum()
            targets = torch.zeros(nB, num_steps+1).long().cuda()
            start_id = 0
            for i in range(nB):
                targets[i][1:1+text_length.data[i]] = text.data[start_id:start_id+text_length.data[i]]+1
                start_id = start_id+text_length.data[i]
            targets = Variable(targets.transpose(0,1).contiguous())

            output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
            hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))
            max_locs = torch.zeros(num_steps, nB)
            max_vals = torch.zeros(num_steps, nB)
            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets[i])
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                output_hiddens[i] = hidden
                if self.processed_batches % 500 == 0:
                    max_val, max_loc = alpha.data.max(1)
                    max_locs[i] = max_loc.cpu()
                    max_vals[i] = max_val.cpu()
            if self.processed_batches % 500 == 0:
                print('max_locs', list(max_locs[0:text_length.data[0],0]))
                print('max_vals', list(max_vals[0:text_length.data[0],0]))
            new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data))
            b = 0
            start = 0
            for length in text_length.data:
                new_hiddens[start:start+length] = output_hiddens[0:length,b,:]
                start = start + length
                b = b + 1
            probs = self.generator(new_hiddens)
            return probs
        else:
            self.processed_batches = self.processed_batches + 1
            nT = feats.size(0)
            nB = feats.size(1)
            nC = feats.size(2)

            hidden_size = self.hidden_size
            input_size = self.input_size
            assert(input_size == nC)
            
            num_steps = text_length.data.max()
            num_labels = text_length.data.sum()

            hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))

            targets_temp = Variable(torch.zeros(nB).long().cuda().contiguous())
            probs = Variable(torch.zeros(nB*num_steps, self.num_classes).cuda())

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets_temp)
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                hidden2class = self.generator(hidden)
                probs[i*nB:(i+1)*nB] = hidden2class
                _, targets_temp = hidden2class.max(1)
                targets_temp += 1
            probs = probs.view(num_steps, nB, self.num_classes).permute(1, 0, 2).contiguous()
            probs = probs.view(-1, self.num_classes).contiguous()
            probs_res = Variable(torch.zeros(num_labels, self.num_classes).type_as(feats.data))
            b = 0
            start = 0
            for length in text_length.data:
                probs_res[start:start+length] = probs[b*num_steps:b*num_steps+length]
                start = start + length
                b = b + 1
            return probs_res

class ASRN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, BidirDecoder=False):
        super(ASRN, self).__init__()
        assert imgH % 16 == 0, 'imgH must be a multiple of 16'
        self.cnn = nn.Sequential(
                        nn.Conv2d(nc, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2), # 64x16x50
                        nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2), # 128x8x25
                        nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), # 256x8x25
                        nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 256x4x25
                        nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), # 512x4x25
                        nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 512x2x25
                        nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
                        ) # 512x1x25
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh),
            )

        self.BidirDecoder = BidirDecoder
        if self.BidirDecoder:
            self.attentionL2R = Attention(nh, nh, nclass, 256)
            self.attentionR2L = Attention(nh, nh, nclass, 256)
        else:
            self.attention = Attention(nh, nh, nclass, 256)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        for layer in self.cnn:
            classname = layer.__class__.__name__
            if classname.find('Conv') != -1:
                layer.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)

    def forward(self, input, length, text, text_rev, test=False):
        # conv features
        conv = self.cnn(input)
        
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1).contiguous()  # [w, b, c]

        # rnn features
        rnn = self.rnn(conv)

        if self.BidirDecoder:
            output0 = self.attentionL2R(rnn, length, text, test)
            output1 = self.attentionR2L(rnn, length, text_rev, test)
            return output0, output1
        else:
            output = self.attention(rnn, length, text, test)
            return output
