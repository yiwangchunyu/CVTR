import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.transforms import ToTensor

import crnn_ctc, utils_ctc


def process_img(img_path, imgW=160, imgH=32):
    toTensor = ToTensor()
    image = Image.open(img_path).convert('L')
    image = image.transpose(Image.ROTATE_90)
    image = image.resize((imgW, imgH), Image.BILINEAR)
    image = toTensor(image)
    image.sub_(0.5).div_(0.5)
    return image

def predict(pth='expr/crnn_best.pth', img_path='', imgW=160,imgH=32, display=True):
    with open('../data/images/alphabet.txt',encoding='utf-8') as f:
        alphabet=f.read()
    nclass=len(alphabet)+1
    converter = utils_ctc.strLabelConverter(alphabet)
    batch_size=1

    image=process_img(img_path, imgW=imgW, imgH=imgH)

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
    if display:
        image = Image.open(img_path).convert('RGB')
        raw_preds_size=len(raw_preds)
        image = image.resize((image.size[0]*raw_preds_size,image.size[1]*raw_preds_size), Image.BILINEAR)
        raw_w=sim_w=image.size[0]//2
        image_dsp = Image.new('RGB',(image.size[0]+raw_w+sim_w,image.size[1]),ImageColor.getrgb('cornsilk'))
        image_dsp = np.array(image_dsp)
        image_dsp[:,image.size[0]+raw_w:,:]=ImageColor.getrgb('bisque')
        image_dsp[:image.size[1],:image.size[0],:]=np.array(image)
        image_dsp = Image.fromarray(image_dsp)
        draw=ImageDraw.Draw(image_dsp)
        rect_h=image_dsp.size[1]//raw_preds_size
        for i in range(raw_preds_size):
            draw.rectangle((0,i*rect_h,image.size[0],(i+1)*rect_h),outline='red',width=10)
            draw.text((image.size[0]+raw_w//3,i*rect_h),raw_preds[i],fill='red',font=ImageFont.truetype('../data/fonts/simkai.ttf', rect_h))
            if (i==0 and raw_preds[i]!='-') or (i>0 and raw_preds[i]!='-' and raw_preds[i]!=raw_preds[i-1]):
                draw.text((image.size[0]+raw_w+sim_w//3, i * rect_h), raw_preds[i], fill='red',font=ImageFont.truetype('../data/fonts/simkai.ttf', rect_h))
        image_dsp.show()
        print(len(raw_preds), raw_preds)
        print(len(sim_preds), sim_preds)
    return sim_preds

if __name__=='__main__':
    predict(img_path='../data/images/test/test_5.png')