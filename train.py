from LoadUCF101DataByTorch import trainset_loader, testset_loader
from model import LRCN
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



EPOCH = 10
LEARNING_RATE = 0.003
MOMENTUM = 0.9
GAMMA = 0.5
STEP_SIZE = 1



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)



model = LRCN().to(device)



optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM
)
# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=STEP_SIZE,
#     gamma=GAMMA
# )



def save_checkpoint(path, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)



def train(epoch):
    iteration = 0
    model.train()
    loss_plt=[]

    model.train()

    for i in range(epoch):
        # print('current lr', scheduler.get_last_lr())
        for index, data in enumerate(trainset_loader):
            video_clips, label = data

            video_clips = video_clips.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(video_clips)

            loss = F.cross_entropy(output, label)

            loss_plt.append(loss.item())

            loss.backward()
            optimizer.step()

            iteration += 1

            print("Epoch:", i, "/", epoch-1, "\tIteration:", index, "/", len(trainset_loader)-1, "\tLoss: " + str(loss.item()))
            with open('log.txt', 'a') as f:
                f.write("Epoch: " + str(i) + "/" + str(epoch-1) + "\tIteration:" + str(index) + "/" + str(len(trainset_loader)-1) + "\tLoss: " + str(loss.item()) + "\n")

        save_checkpoint('model/checkpoint-%i.pth' % iteration, model, optimizer)

        test(i)

        # scheduler.step()

    save_checkpoint('model/checkpoint-%i.pth' % iteration, model, optimizer)

    plt.figure()
    plt.plot(loss_plt)
    plt.title('Loss')
    plt.xlabel('Iteration')
    plt.ylabel('')
    plt.show()



def test(i_epoch):
    model.eval()

    correct = 0

    with torch.no_grad():
        for index, data in enumerate(testset_loader):
            video_clips, label = data

            video_clips = video_clips.to(device)
            label = label.to(device)

            output = model(video_clips)

            max_value, max_index = output.max(1, keepdim=True)
            correct += max_index.eq(label.view_as(max_index)).sum().item()

    print("Accuracy: " + str(correct * 1.0 * 100 / len(testset_loader.dataset)))
    with open('log.txt', 'a') as f:
        f.write("Epoch " + str(i_epoch) + "'s Accuracy: " + str(correct * 1.0 * 100 / len(testset_loader.dataset)) + "\n")


if __name__ == '__main__':
    train(EPOCH)
