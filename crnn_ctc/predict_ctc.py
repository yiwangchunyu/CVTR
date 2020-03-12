import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from crnn_ctc import crnn_ctc, utils_ctc


def process_img(img_path):
    toTensor = ToTensor()
    image = Image.open(img_path).convert('L')
    image = image.transpose(Image.ROTATE_90)
    image = image.resize((160, 32), Image.BILINEAR)
    image = toTensor(image)
    image.sub_(0.5).div_(0.5)
    return image

def predict(pth='expr/crnn_best.pth', img_path=''):
    with open('../data/images.alphabet.txt') as f:
        alphabet=f.read()
    nclass=len(alphabet)+1
    converter = utils_ctc.strLabelConverter(alphabet)
    transformer=''
    batch_size=1

    image=process_img(img_path)

    image = image.view(1, *image.size())
    image = Variable(image)

    crnn = crnn_ctc.CRNN(32, 1, nclass, 256)
    crnn.load_state_dict(torch.load(pth, map_location='cpu'))
    crnn.eval()
    preds = crnn(image)

    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)

    print(raw_preds)
    predict(sim_preds)


if __name__=='__main__':
    predict(img_path='data/images/test/test_0.png')