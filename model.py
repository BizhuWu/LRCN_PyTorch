import torch.nn as nn
import torch
import torchvision.models as models
import config




class LRCN(nn.Module):
    def __init__(self):
        super(LRCN, self).__init__()

        # define the CNN part
        self.featureExtractor = models.alexnet(pretrained=True)
        # remove fc7
        self.featureExtractor.classifier = nn.Sequential(*list(self.featureExtractor.classifier.children())[:-5])

        # define the lstm part
        self.lstm = nn.LSTM(input_size=config.input_size, hidden_size=config.hidden_size, num_layers=config.num_of_layers, dropout=0.9, batch_first=True)
        # define a linear layer
        self.linearLayer = nn.Linear(config.hidden_size, config.classNum)


    def forward(self, video_clip):
        # video clip's dimension: [B, C, T, H, W]

        # frameFeatures' dimension: [B, T, CNN's output dimension(4096)]
        # it is used to store all frame's feature
        frameFeatures = torch.empty(size=(video_clip.size()[0], video_clip.size()[2], config.input_size), device='cuda')


        for t in range(0, video_clip.size()[2]):
            frame = video_clip[:, :, t, :, :]
            frame_feature = self.featureExtractor(frame)
            # print(frame_feature.shape)
            frameFeatures[:, t, :] = frame_feature


        # x is the output of lstm：(batch, seq_len, input_size)
        x, _ = self.lstm(frameFeatures)

        batch_size, seq_length, hidden_size = x.size()
        x = x.reshape(-1, hidden_size)

        x = self.linearLayer(x)

        # x：(batch, seq_length, class_Num)
        x = x.view(batch_size, seq_length, -1)              # (seq_length, batch_size, classNum)

        # x：(batch, class_Num)
        x = torch.mean(x, dim=1)

        return x



if __name__ == '__main__':
    model = LRCN()
    frames = torch.rand(config.BATCH_SIZE, 3, config.seq_length, 227, 227)
    output = model(frames)

    print(output.size())
