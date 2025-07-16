# from torch.functional import Tensor
# from torch.utils.data import DataLoader
import torch
import numpy as np

def denorm(img):

    np_input = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        np_input = True

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    img_denorm = (img*std[:,None,None]) + mean[:,None,None]

    if np_input:
        img_denorm = np.clip(img_denorm.numpy(), 0, 1)
    else:
        img_denorm = torch.clamp(img_denorm, 0, 1)

    return img_denorm


def norm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    return (img - mean[:,None,None]) / std[:,None,None]


def fast_iou_curve(p, g):
    
    g = g[p.sort().indices]
    p = torch.sigmoid(p.sort().values)
    
    scores = []
    vals = np.linspace(0, 1, 50)

    for q in vals:

        n = int(len(g) * q)

        valid = torch.where(p > q)[0]
        if len(valid) > 0:
            n = int(valid[0])
        else:
            n = len(g)

        fn = g[:n].sum()
        tn = n - fn
        tp = g[n:].sum()
        fp = len(g) - n - tp

        iou = tp / (tp + fn + fp)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        scores += [iou]
        
    return vals, scores


def fast_rp_curve(p, g):
    
    g = g[p.sort().indices]
    p = torch.sigmoid(p.sort().values)
    
    precisions, recalls = [], []
    vals = np.linspace(p.min(), p.max(), 250)

    for q in p[::100000]:

        n = int(len(g) * q)

        valid = torch.where(p > q)[0]
        if len(valid) > 0:
            n = int(valid[0])
        else:
            n = len(g)

        fn = g[:n].sum()
        tn = n - fn
        tp = g[n:].sum()
        fp = len(g) - n - tp

        iou = tp / (tp + fn + fp)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        precisions += [precision]
        recalls += [recall]
        
    return recalls, precisions


# Image processing

def img_preprocess(batch, blur=0, grayscale=False, center_context=None, rect=False, rect_color=(255,0,0), rect_width=2, 
                   brightness=1.0, bg_fac=1, colorize=False, outline=False, image_size=224,whitelize=False):
    import cv2

    rw = rect_width

    out = []
    for img, mask in zip(batch[1], batch[2]):

        img = img.cpu() if isinstance(img, torch.Tensor) else torch.from_numpy(img)
        mask = mask.cpu() if isinstance(mask, torch.Tensor) else torch.from_numpy(mask)
        
        img *= brightness
        img_bl = img
        if blur > 0: # best 5
            img_bl = torch.from_numpy(cv2.GaussianBlur(img.permute(1,2,0).numpy(), (15, 15), blur)).permute(2,0,1)
        
        if grayscale:
            img_bl = img_bl[1][None]
        
        #img_inp = img_ratio*img*mask + (1-img_ratio)*img_bl
        # img_inp = img_ratio*img*mask + (1-img_ratio)*img_bl * (1-mask)
        if whitelize:
            img_inp = img * mask + (1-(bg_fac) * (1-img_bl)) * (1 - mask)
        else:
            img_inp = img*mask + (bg_fac) * img_bl * (1-mask)

        if rect:
            _, bbox = crop_mask(img, mask, context=0.1)
            img_inp[:, bbox[2]: bbox[3], max(0, bbox[0]-rw):bbox[0]+rw] = torch.tensor(rect_color)[:,None,None]
            img_inp[:, bbox[2]: bbox[3], max(0, bbox[1]-rw):bbox[1]+rw] = torch.tensor(rect_color)[:,None,None]
            img_inp[:, max(0, bbox[2]-1): bbox[2]+rw, bbox[0]:bbox[1]] = torch.tensor(rect_color)[:,None,None]
            img_inp[:, max(0, bbox[3]-1): bbox[3]+rw, bbox[0]:bbox[1]] = torch.tensor(rect_color)[:,None,None]


        if center_context is not None:
            img_inp = object_crop(img_inp, mask, context=center_context, image_size=image_size)

        if colorize:
            img_gray = denorm(img)
            img_gray = cv2.cvtColor(img_gray.permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY)
            img_gray = torch.stack([torch.from_numpy(img_gray)]*3)
            img_inp = torch.tensor([1,0.2,0.2])[:,None,None] * img_gray * mask + bg_fac * img_gray * (1-mask)
            img_inp = norm(img_inp)

        if outline:
            cont = cv2.findContours(mask.byte().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline_img = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(outline_img, cont[0], -1, thickness=5, color=(255, 255, 255))
            outline_img = torch.stack([torch.from_numpy(outline_img)]*3).float() / 255.
            img_inp = torch.tensor([1,0,0])[:,None,None] *  outline_img + denorm(img_inp) * (1- outline_img)
            img_inp = norm(img_inp)

        out += [img_inp]

    return torch.stack(out)
def object_crop(img, mask, context=0.0, square=False, image_size=224):
    img_crop, bbox = crop_mask(img, mask, context=context, square=square)
    img_crop = pad_to_square(img_crop, channel_dim=0)
    img_crop = torch.nn.functional.interpolate(img_crop.unsqueeze(0), (image_size, image_size)).squeeze(0)
    return img_crop
    

def crop_mask(img, mask, context=0.0, square=False):
    
    assert img.shape[1:] == mask.shape
    
    bbox = [mask.max(0).values.argmax(), mask.size(0) - mask.max(0).values.flip(0).argmax()]
    bbox += [mask.max(1).values.argmax(), mask.size(1) - mask.max(1).values.flip(0).argmax()]
    bbox = [int(x) for x in bbox]
    
    width, height = (bbox[3] - bbox[2]), (bbox[1] - bbox[0])

    # square mask
    if square:
        bbox[0] = int(max(0, bbox[0] - context * height))
        bbox[1] = int(min(mask.size(0), bbox[1] + context * height))
        bbox[2] = int(max(0, bbox[2] - context * width))
        bbox[3] = int(min(mask.size(1), bbox[3] + context * width))

        width, height = (bbox[3] - bbox[2]), (bbox[1] - bbox[0])
        if height > width:
            bbox[2] = int(max(0, (bbox[2] - 0.5*height)))
            bbox[3] = bbox[2] + height
        else:
            bbox[0] = int(max(0, (bbox[0] - 0.5*width)))
            bbox[1] = bbox[0] + width
    else:
        bbox[0] = int(max(0, bbox[0] - context * height))
        bbox[1] = int(min(mask.size(0), bbox[1] + context * height))
        bbox[2] = int(max(0, bbox[2] - context * width))
        bbox[3] = int(min(mask.size(1), bbox[3] + context * width))

    width, height = (bbox[3] - bbox[2]), (bbox[1] - bbox[0])
    img_crop = img[:, bbox[2]: bbox[3], bbox[0]: bbox[1]]
    return img_crop, bbox


def pad_to_square(img, channel_dim=2, fill=0):
    """


    add padding such that a squared image is returned """
    
    from torchvision.transforms.functional import pad

    if channel_dim == 2:
        img = img.permute(2, 0, 1)
    elif channel_dim == 0:
        pass
    else:
        raise ValueError('invalid channel_dim')

    h, w = img.shape[1:]
    pady1 = pady2 = padx1 = padx2 = 0

    if h > w:
        padx1 = (h - w) // 2
        padx2 = h - w - padx1
    elif w > h:
        pady1 = (w - h) // 2
        pady2 = w - h - pady1

    img_padded = pad(img, padding=(padx1, pady1, padx2, pady2), padding_mode='constant')

    if channel_dim == 2:
        img_padded = img_padded.permute(1, 2, 0)

    return img_padded


# qualitative

def split_sentence(inp, limit=9):
    t_new, current_len = [], 0
    for k, t in enumerate(inp.split(' ')):
        current_len += len(t) + 1
        t_new += [t+' ']
        # not last
        if current_len > limit and k != len(inp.split(' ')) - 1:
            current_len = 0
            t_new += ['\n']

    t_new = ''.join(t_new)
    return t_new
    

from matplotlib import pyplot as plt


def plot(imgs, *preds, labels=None, scale=1, cmap=plt.cm.magma, aps=None, gt_labels=None, vmax=None):
    
    row_off = 0 if labels is None else 1
    _, ax = plt.subplots(len(imgs) + row_off, 1 + len(preds), figsize=(scale * float(1 + 2*len(preds)), scale * float(len(imgs)*2)))
    [a.axis('off') for a in ax.flatten()]
    
    if labels is not None:
        for j in range(len(labels)):
            t_new = split_sentence(labels[j], limit=6)
            ax[0, 1+ j].text(0.5, 0.1, t_new, ha='center', fontsize=3+ 10*scale)


    for i in range(len(imgs)):
        ax[i + row_off,0].imshow(imgs[i])
        for j in range(len(preds)):
            img = preds[j][i][0].detach().cpu().numpy()

            if gt_labels is not None and labels[j] == gt_labels[i]:
                print(j, labels[j], gt_labels[i])
                edgecolor = 'red'
                if aps is not None:
                    ax[i + row_off, 1 + j].text(30, 70, f'AP: {aps[i]:.3f}', color='red', fontsize=8)
            else:
                edgecolor = 'k'

            rect = plt.Rectangle([0,0], img.shape[0], img.shape[1], facecolor="none", 
                                 edgecolor=edgecolor, linewidth=3)
            ax[i + row_off,1 + j].add_patch(rect)

            if vmax is None:
                this_vmax = 1 
            elif vmax == 'per_prompt':
                this_vmax = max([preds[j][_i][0].max() for _i in range(len(imgs))])
            elif vmax == 'per_image':
                this_vmax = max([preds[_j][i][0].max() for _j in range(len(preds))])

            ax[i + row_off,1 + j].imshow(img, vmin=0, vmax=this_vmax, cmap=cmap)

    
            # ax[i,1 + j].imshow(preds[j][i][0].detach().cpu().numpy(), vmin=preds[j].min(), vmax=preds[j].max())
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)


import cv2
import numpy as np
import torch
import torchvision.transforms as T


def apply_mask_and_noise(image_tensor, mask_polygons, noise_std=0.1):
    """
    处理图像：
    1. 将多边形掩码转换为二值掩码
    2. 保留目标区域，背景区域变白
    3. 添加高斯噪声

    参数:
    image_tensor: 输入图像张量 (1, 3, H, W) 值域[0,1]
    mask_polygons: 多边形列表 [每个多边形为[N,2]数组]
    noise_std: 高斯噪声标准差

    返回:
    处理后的图像张量
    """
    # 设备信息
    device = image_tensor.device

    # 图像尺寸
    _, _, H, W = image_tensor.shape

    # 创建空白掩码
    mask_np = np.zeros((H, W), dtype=np.uint8)

    # 将多边形绘制到掩码上
    for polygon in mask_polygons:
        # 将多边形坐标整形为OpenCV格式
        pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask_np, [pts], color=1)

    # 转换为PyTorch张量
    mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).to(device)
    mask_tensor = mask_tensor[None, None, :, :]  # 增加维度 [1,1,H,W]

    # 变白处理 (假设图像归一化到[0,1])
    white_value = 1.0  # 归一化后的白色值
    processed_img = image_tensor * mask_tensor + white_value * (1 - mask_tensor)

    # 添加高斯噪声
    noise = torch.randn_like(processed_img) * noise_std
    noisy_img = processed_img + noise

    # 裁剪到合法范围 [0,1]
    return torch.clamp(noisy_img, 0.0, 1.0)
import torch
import numpy as np
if __name__ == "__main__":
    # impath=r'E:\odinw\NorthAmericaMushrooms\NorthAmericaMushrooms\North American Mushrooms.v1-416x416.coco\train\chanterelle_03_jpg.rf.026cfdfaa9d1c12724e849b54e2c0a63.jpg'
    # import matplotlib.pyplot as plt
    # from skimage import io
    # im=io.imread(impath)
    # scale=0.5
    # im=im*scale+0*(1-scale)
    # plt.imshow(im)
    # plt.show()
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import io, img_as_float, img_as_ubyte
    from skimage.filters import gaussian

    # 读取图像
    impath = r'E:\odinw\plant\plantdoc\416x416\train\4e55f05d945952cfad9ce97ca444c389_jpg.rf.062d5a06904658dc434891bd3e757eac.jpg'#r'E:\odinw\openPoetryVis\openPoetryVision\512x512\train\1_jpg.rf.df24968b057326895bffc5a5b8a6267c.jpg'#erican Mushrooms.v1-416x416.coco\train\chanterelle_03_jpg.rf.026cfdfaa9d1c12724e849b54e2c0a63.jpg'
    # impath=r'E:\odinw\plant\plantdoc\416x416\train\dsc5270_jpg.rf.954a7cf1041fd817e3bc26e9eb937d8a.jpg'
    im = io.imread(impath)

    # 定义边界框参数
    bbox = [         ]
    bg_fac = 0.5  # 背景变暗因子
    blur_sigma = 3  # 高斯模糊强度

    # 计算边界框的坐标范围
    x_min = int(round(bbox[0]))
    y_min = int(round(bbox[1]))
    width = bbox[2]
    height = bbox[3]
    x_max = int(round(x_min + width))
    y_max = int(round(y_min + height))

    # 创建蒙版（True表示保留区域）
    mask = np.zeros(im.shape[:2], dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True

    # 将图像转换为浮点数以便处理
    im_float = img_as_float(im)

    # 对整张图像应用高斯模糊
    blurred = gaussian(im_float, sigma=blur_sigma, channel_axis=-1)

    # 创建结果图像，外部区域变暗并模糊，内部保持原样
    result = blurred.copy()
    result[~mask] *= bg_fac  # 外部区域变暗
    result[mask] = im_float[mask]  # 恢复内部区域

    # 转换回uint8类型并显示结果
    result_im = img_as_ubyte(result)
    io.imsave(r'E:\odinw\mush1.jpg',result_im)
    plt.figure(figsize=(10, 10))
    plt.imshow(result_im)
    plt.axis('off')
    plt.show()