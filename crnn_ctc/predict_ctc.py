import os

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.transforms import ToTensor

import crnn_ctc, utils_ctc


def process_img(img_path, imgW=160, imgH=32):
    toTensor = ToTensor()
    image = Image.open(img_path).convert('RGB')
    #图片保持比例转成32*280
    ratio = 280/32
    imageW,imageH=image.size
    target_H=int(imageW*ratio)
    image_32_280=None
    if target_H>imageH:
        #add padding
        image_pad=Image.new('RGB',(imageW,target_H),(255,255,255))
        image_pad=np.array(image_pad)
        image_pad[:imageH,:imageW,:]=np.array(image)
        image_pad = Image.fromarray(image_pad)
        image_32_280=image_pad.resize((32,280),Image.BILINEAR)
        image=image_pad
    else:
        target_W=int(imageH/ratio)
        image_pad = Image.new('RGB', (target_W, imageH), (255, 255, 255))
        pad=(target_W-imageW)
        image_pad = np.array(image_pad)
        image_pad[:imageH, :imageW, :] = np.array(image)
        # pad=(target_W-imageW)//2
        # image_pad = np.array(image_pad)
        # image_pad[:imageH, pad:pad+imageW, :] = np.array(image)
        image_pad = Image.fromarray(image_pad)
        image_32_280 = image_pad.resize((32, 280), Image.BILINEAR)
        image = image_pad
    image = image.convert('L')
    image = image.transpose(Image.ROTATE_90)
    image = image.resize((imgW, imgH), Image.BILINEAR)
    image = toTensor(image)
    image.sub_(0.5).div_(0.5)
    return image,image_32_280

def predict(pth='expr/200/crnn_best_200.pth', img_path='', imgW=160,imgH=32, display=True):
    with open('../data/images/alphabet.txt',encoding='utf-8') as f:
        alphabet=f.read()
    nclass=len(alphabet)+1
    res_path='expr/{}'.format(nclass-1)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    converter = utils_ctc.strLabelConverter(alphabet)
    batch_size=1

    image, image_32_280=process_img(img_path, imgW=imgW, imgH=imgH)

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
        image = image_32_280
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
            draw.rectangle((0,i*rect_h,image.size[0],(i+1)*rect_h),outline='red',width=5)
            draw.text((image.size[0]+raw_w//3,i*rect_h),raw_preds[i],fill='red',font=ImageFont.truetype('../data/fonts/simkai.ttf', rect_h))
            if (i==0 and raw_preds[i]!='-') or (i>0 and raw_preds[i]!='-' and raw_preds[i]!=raw_preds[i-1]):
                draw.text((image.size[0]+raw_w+sim_w//3, i * rect_h), raw_preds[i], fill='red',font=ImageFont.truetype('../data/fonts/simkai.ttf', rect_h))
        image_dsp.show()
        image_dsp.save('{0}/{1}_res.png'.format(res_path,img_path.split('/')[-1].split('.')[0]))
        print(len(raw_preds), raw_preds)
        print(len(sim_preds), sim_preds)
    return sim_preds

if __name__=='__main__':
    predict(img_path='../data/images/test/test_00000000018.png')