import os

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm

import crnn_ctc, utils_ctc


def process_img(img_path, imgW=160, imgH=32):
    padding_color=(255,255,255)
    image = Image.open(img_path).convert('RGB')
    #图片保持比例转成32*280
    ratio = 280/32
    res=[]
    def crop(img_crop):
        res=[]


        return res

    def process_single(image_single):
        image_singleW, image_singleH = image_single.size
        target_singleH = int(image_singleW * ratio)
        image_32_280 = None
        if target_singleH > image_singleH:
            # add padding
            image_pad = Image.new('RGB', (image_singleW, target_singleH), padding_color)
            image_pad = np.array(image_pad)
            image_pad[:image_singleH, :image_singleW, :] = np.array(image_single)
            image_pad = Image.fromarray(image_pad)
            image_32_280 = image_pad.resize((32, 280), Image.BILINEAR)
        else:
            target_single_W = int(image_singleH / ratio)
            image_pad = Image.new('RGB', (target_single_W, image_singleH), padding_color)
            image_pad = np.array(image_pad)
            image_pad[:image_singleH, :image_singleW, :] = np.array(image_single)
            image_pad = Image.fromarray(image_pad)
            image_32_280 = image_pad.resize((32, 280), Image.BILINEAR)
        return image_32_280

    imageW,imageH=image.size
    if imageH/imageW<=10:
        image_32_280=process_single(image)
        res.append(image_32_280)
    else:
        #需要切割成多张图片
        target_H = int(imageW * ratio)
        times = imageH // target_H +1
        if imageH%target_H==0:
            times-=1
        eachH=imageH//times
        for i in range(times):
            slice=image.crop((0,i*eachH,imageW,(i+1)*eachH))
            image_32_280 = process_single(slice)
            res.append(image_32_280)
        pass
    return res

def resizeNormalize(image,imgW,imgH):
    toTensor = ToTensor()
    image = image.convert('L')
    image = image.transpose(Image.ROTATE_90)
    image = image.resize((imgW, imgH), Image.BILINEAR)
    image = toTensor(image)
    image.sub_(0.5).div_(0.5)
    return image

def predict(pth='expr/200/crnn_best_200.pth', img_path='', imgW=160,imgH=32, display=True):
    with open('../data/images/alphabet.txt',encoding='utf-8') as f:
        alphabet=f.read()
    nclass=len(alphabet)+1
    res_path='expr/{}'.format(nclass-1)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    converter = utils_ctc.strLabelConverter(alphabet)
    batch_size=1
    text=''
    raw_text=''
    image_long=None

    images=process_img(img_path, imgW=imgW, imgH=imgH)
    for image in images:
        image_tensor=resizeNormalize(image,imgW,imgH)
        image_tensor_batch1=image_tensor.view(1, *image_tensor.size())
        image_variable=Variable(image_tensor_batch1)

        crnn = crnn_ctc.CRNN(32, 1, nclass, 256)
        crnn.load_state_dict(torch.load(pth, map_location='cpu'))
        crnn.eval()
        preds = crnn(image_variable)

        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)
        raw_text+=raw_preds
        text+=sim_preds

        if image_long is None:
            image_long=np.array(image)
        else:
            image_long=np.vstack((image_long,image))

    if display:
        image=Image.fromarray(image_long)
        raw_text_size=len(raw_text)
        image = image.resize((image.size[0]*raw_text_size,image.size[1]*raw_text_size), Image.BILINEAR)
        raw_w=sim_w=image.size[0]//2
        image_dsp = Image.new('RGB',(image.size[0]+raw_w+sim_w,image.size[1]),ImageColor.getrgb('cornsilk'))
        image_dsp = np.array(image_dsp)
        image_dsp[:,image.size[0]+raw_w:,:]=ImageColor.getrgb('bisque')
        image_dsp[:image.size[1],:image.size[0],:]=np.array(image)
        image_dsp = Image.fromarray(image_dsp)
        draw=ImageDraw.Draw(image_dsp)
        rect_h=image_dsp.size[1]//raw_text_size
        for i in range(raw_text_size):
            draw.rectangle((0,i*rect_h,image.size[0],(i+1)*rect_h),outline='red',width=5)
            draw.text((image.size[0]+raw_w//3,i*rect_h),raw_text[i],fill='red',font=ImageFont.truetype('../data/fonts/simkai.ttf', rect_h))
            if (i==0 and raw_text[i]!='-') or (i>0 and raw_text[i]!='-' and raw_text[i]!=raw_text[i-1]):
                draw.text((image.size[0]+raw_w+sim_w//3, i * rect_h), raw_text[i], fill='red',font=ImageFont.truetype('../data/fonts/simkai.ttf', rect_h))
        image_dsp.show()
        image_dsp.save('{0}/{1}_res.png'.format(res_path,img_path.split('/')[-1].split('.')[0]))
        print(len(raw_text), raw_text)
        print(len(text), text)
    return raw_text,text

def predictBatch(root='../data/images/test/'):
    img_names=os.listdir(root)
    for img_name in tqdm(img_names):
        predict(pth='expr/800/crnn_best_800.pth', img_path=root+img_name)

if __name__=='__main__':
    predictBatch()
    # print(predict(pth='expr/800/crnn_best_800.pth',img_path='../data/images/test/test_0000000009.png'))