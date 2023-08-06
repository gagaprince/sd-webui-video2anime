from tagger.tagger import utils
import os.path
import time

import cv2
import numpy
from PIL import Image, ImageOps

import subprocess

import modules.images
import os
import sys
from modules.shared import opts, state
from modules import shared, sd_samplers, processing, images
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, process_images, Processed
import rembg
from scripts.m2a_config import m2a_outpath_samples, m2a_output_dir, m2a_eb_output_dir

import time
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor

_kernel = None

def refresh_interrogators():
    utils.refresh_interrogators()
    print('utils.interrogators', utils.interrogators)

def getTagsFromImage(image, isImage, invoke_tagger_val, common_invoke_tagger):
    key = 'wd14-vit-v2-git'
    interrogator = utils.interrogators[key]
    img_pil = image
    if not isImage:
        img_pil = Image.fromarray(image)
    print('img_pil', img_pil)
    ratings, tags = interrogator.interrogate(img_pil)

    tagsSelect = []

    for k in tags:
        tagV = tags[k]
        if tagV > invoke_tagger_val and k != 'realistic':
            tagsSelect.append(k)

    # ret = '((masterpiece)),best quality,ultra-detailed,smooth,best illustrationbest, shadow,photorealistic,hyperrealistic,backlighting,' + ','.join(tagsSelect)
    ret = common_invoke_tagger + ','.join(tagsSelect)
    return ret

def getTagsFromImages(images, invoke_tagger_val, common_invoke_tagger):
    length = len(images)
    image = images[ length // 2 ]
    return getTagsFromImage(image, False, invoke_tagger_val, common_invoke_tagger)


def rembg_mov(image, return_mask=False):
    model='u2net_human_seg'
    only_mask = return_mask
    alpha_matting = False
    alpha_matting_foreground_threshold = 240
    alpha_matting_background_threshold = 10
    alpha_matting_erode_size = 10
    return rembg.remove(
        image,
        session=rembg.new_session(model),
        only_mask=only_mask,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=alpha_matting_background_threshold,
        alpha_matting_erode_size=alpha_matting_erode_size,
    )

def rembg_mov_cv2(cvImg, return_mask=False):
    img = Image.fromarray(cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB), 'RGB')
    img = ImageOps.exif_transpose(img)
    img = rembg_mov(img, return_mask)
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def get_movie_all_images(file, fps_child, fps_parent):
    if file is None:
        return None
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    fpsscale = 1
    skip = 0
    if fps > 30: #大于30帧的要等比缩 因为太耗时了
        fpsscale = int(fps / 30)
        frames = fps / fpsscale / fps_parent * fps_child
    else:
        frames = fps / fps_parent * fps_child
    count = 1
    fs = 1
    image_list = []
    while (True):
        flag, frame = cap.read()
        if not flag:
            break
        else:
            if skip == 0: #取parent帧 得 child帧
                if fs % fps_parent < fps_child:
                    image_list.append(frame)
                    count += 1
            elif fs % skip == 0:
                image_list.append(frame)
                count += 1
        fs += 1
    cap.release()
    print('old list len:',len(image_list))
    image_list = image_list[::fpsscale]
    print('new list len:', len(image_list))
    return image_list, frames

def images_to_video(images, frames, mode, w, h, out_path):
    fourcc = cv2.VideoWriter_fourcc(*mode)
    video = cv2.VideoWriter(out_path, fourcc, frames, (w, h))
    for image in images:
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        video.write(img)
    video.release()
    return out_path

def imageFiles_to_video(imageFiles, fps, mode, w, h, out_path):
    print('imageFiles:', imageFiles)
    print('fps:', fps)
    print('mode:', mode)
    print('w,h:',w,h)
    print('out_path:',out_path)
    if len(imageFiles) == 0:
        return
    fourcc = cv2.VideoWriter_fourcc(*mode)
    size = (w, h)
    vid = cv2.VideoWriter(os.path.join(os.getcwd(),out_path), fourcc, fps, size)
    for imageFile in imageFiles:
        imageFile = os.path.join(os.getcwd(),imageFile)
        print('imageFile:', imageFile)
        img = cv2.imread(imageFile, 1)
        vid.write(img)

    vid.release()

    return out_path


def mkWorkDir():
    if not os.path.exists(m2a_eb_output_dir):
        os.makedirs(m2a_eb_output_dir, exist_ok=True)
    now = time.time()
    now = int(now)
    workDir = os.path.join(m2a_eb_output_dir, str(now))
    keyDir = os.path.join(workDir,'key')
    videoDir = os.path.join(workDir,'video')
    outTmpDir = os.path.join(workDir,'tmpout')
    outDir = os.path.join(workDir, 'out')
    maskDir = os.path.join(workDir,'mask')
    os.makedirs(workDir, exist_ok=True)
    os.makedirs(keyDir, exist_ok=True)
    os.makedirs(videoDir, exist_ok=True)
    os.makedirs(outTmpDir, exist_ok=True)
    os.makedirs(outDir, exist_ok=True)
    os.makedirs(maskDir, exist_ok=True)
    return workDir,keyDir,videoDir,outDir, outTmpDir,maskDir

def corpImg(originImg, w, h):
    oldh = originImg.shape[0]
    oldw = originImg.shape[1]
    print(oldw, oldh)

    corpImg = []
    if int(oldw * h / oldh) > w:
        print('需要截取w')
        needw = w * oldh / h
        beginX = int((oldw - needw) / 2)
        endX = int(beginX + needw)
        beginY = 0
        endY = oldh
        corpImg = originImg[beginY:endY, beginX:endX]

    elif int(oldw * h / oldh) < w:
        print('需要截取h')
        needY = h * oldw / w
        beginY = int((oldh - needY) / 2)
        endY = int(beginY + needY)
        beginX = 0
        endX = oldw
        print('beginx:endx, beginY:endY', beginX,endX,beginY,endY)
        corpImg = originImg[beginY:endY, beginX:endX]
    else:
        corpImg = originImg

    corpImg = cv2.resize(corpImg, (w, h))
    return corpImg

def video2imgs(videoPath, imgPath, max_frames, needCorp, w=720,h=1280, maskDir='', isNotNormal=False):
    if not os.path.exists(imgPath):
        os.makedirs(imgPath, exist_ok=True)             # 目标文件夹不存在，则创建
    cap = cv2.VideoCapture(videoPath)    # 获取视频
    judge = cap.isOpened()                 # 判断是否能打开成功
    fps = cap.get(cv2.CAP_PROP_FPS)      # 帧率，视频每秒展示多少张图片
    count = 0                            # 用于统计保存的图片数量
    imgPaths = []
    imgDataList = []

    while(judge):
        if state.interrupted:
            break
        flag, frame = cap.read()         # 读取每一张图片 flag表示是否读取成功，frame是图片
        if not flag:
            print("Process finished!")
            break
        else:
            imgname = str(count)+".png"
            onePath = os.path.join(imgPath,imgname)

            print(onePath)

            if needCorp:
                frame = corpImg(frame,w,h)
            if isNotNormal:
                mask = rembg_mov_cv2(frame, True)
                maskPath = os.path.join(maskDir,imgname)
                cv2.imwrite(maskPath, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            imgDataList.append(frame)
            cv2.imwrite(onePath, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            imgPaths.append(onePath)
            count += 1
        if max_frames != -1 and count >= max_frames:
            break
    cap.release()

    return [imgPaths, imgDataList,fps]


def mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the mean average distance in pixel values between `left` and `right`.
    Both `left and `right` should be 2 dimensional 8-bit images of the same shape.
    """
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)

def estimated_kernel_size(frame_width: int, frame_height: int) -> int:
    """Estimate kernel size based on video resolution."""
    size: int = 4 + round(math.sqrt(frame_width * frame_height) / 192)
    if size % 2 == 0:
        size += 1
    return size
def _detect_edges(lum: np.ndarray) -> np.ndarray:
    global _kernel
    """Detect edges using the luma channel of a frame.
    Arguments:
        lum: 2D 8-bit image representing the luma channel of a frame.
    Returns:
        2D 8-bit image of the same size as the input, where pixels with values of 255
        represent edges, and all other pixels are 0.
    """
    # Initialize kernel.
    if _kernel is None:
        kernel_size = estimated_kernel_size(lum.shape[1], lum.shape[0])
        _kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Estimate levels for thresholding.
    sigma: float = 1.0 / 3.0
    median = np.median(lum)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    # Calculate edges using Canny algorithm, and reduce noise by dilating the edges.
    # This increases edge overlap leading to improved robustness against noise and slow
    # camera movement. Note that very large kernel sizes can negatively affect accuracy.
    edges = cv2.Canny(lum, low, high)
    return cv2.dilate(edges, _kernel)
def detect_edges(img_path):
    print('img_path:', img_path)
    im = cv2.imread(img_path)

    hue, sat, lum = cv2.split(cv2.cvtColor( im , cv2.COLOR_BGR2HSV))
    return _detect_edges(lum)

def selectVideoKeyFrame(videoFrames, min_gap, max_gap, th, add_last_frame):
    keys = []
    key_frames = []

    keys.append(0)
    key_frame = videoFrames[0]
    key_frames.append(key_frame)
    gap = 0
    key_edges = detect_edges(key_frame)
    idx = 1
    while idx < len(videoFrames):
        gap += 1
        if gap < min_gap:
            idx += 1
            continue
        videoFrame = videoFrames[idx]
        edges = detect_edges(videoFrame)

        delta = mean_pixel_distance(edges, key_edges)
        print('delta:', delta)

        _th = th * (max_gap - gap) / max_gap

        if _th < delta:
            keys.append(idx)
            key_frame = videoFrame
            key_frames.append(key_frame)
            key_edges = edges
            gap = 0

        idx+=1

    if add_last_frame:
        last_frame = len(videoFrames)-1
        if not last_frame in keys:
            keys.append(last_frame)
            key_frames.append(videoFrames[last_frame])

    return key_frames, keys

def selectKeyFrames(imgs, perNum):
    index=0
    selectImages = []
    selectIndex = []
    for img in imgs:
        if index % perNum == 0:
            selectImages.append(img)
            selectIndex.append(index)
        index += 1
    return [selectImages, selectIndex]

def runEb(ebsynth, keyframe, guide, frame, outPath):
    args = ['cmd','/c',ebsynth, "-style", keyframe, "-guide", guide, frame, "-weight", "4.0", "-output", outPath]

    print('args:', args)

    # args = ['cmd','/c','ls']

    print(' '.join(args))

    out = subprocess.run(args, stderr=subprocess.PIPE)

    # If the command fails or verbose level > 1, print the command output
    if out.returncode != 0:
        print("Ebsynth returned a nonzero exit code:")

# def transWithEb(keyIndexs, keyFrames, videoFrames, outPath):
#     ebsynth=os.path.join(os.getcwd(),'extensions','sd-webui-video2anime','bin','ebsynth.exe')
#     videoImageIdx = 0
#     keyIdx = 0 #keyIndexs的下标
#     keyIndex = keyIndexs[keyIdx] # 关键帧在video中的下标
#
#     sourceFile = ''
#     guide = ''
#     target = ''
#     outTmpFile = ''
#     outFrames = []
#     print('videoFrames:', videoFrames)
#     while videoImageIdx < len(videoFrames)-1:
#         print('videoImageIdx:', videoImageIdx)
#         if videoImageIdx == keyIndex:
#             sourceFile = keyFrames[keyIdx]
#             guide = videoFrames[videoImageIdx]
#             target = videoFrames[videoImageIdx]
#             keyIdx += 1
#             if keyIdx >= len(keyIndexs): # 没有关键帧了给一个超大的值
#                 keyIndex = 999999
#             else:
#                 keyIndex = keyIndexs[keyIdx] # keyIdx = 1  keyIndex =5
#
#             videoImageIdx -= 1
#         elif videoImageIdx < keyIndex-1:
#             sourceFile = outTmpFile
#             guide = videoFrames[videoImageIdx]
#             target = videoFrames[videoImageIdx+1]
#         elif videoImageIdx == keyIndex-1:
#             print('跳过此帧')
#             videoImageIdx += 1
#             continue
#         outTmpFile = os.path.join(outPath,str(videoImageIdx+1)+'.png')
#         cwd = os.getcwd()
#         runEb(ebsynth, os.path.join(cwd,sourceFile), os.path.join(cwd,guide), os.path.join(cwd,target), os.path.join(cwd,outTmpFile))
#         videoImageIdx += 1
#         outFrames.append(outTmpFile)
#
#     return outFrames

def ebTask(options):
    [videoFrames, outPath, keyIndex, ebsynth, cwd, sourceFile, guide, beginKeyIndex, endKeyIndex] = options
    for i in range(beginKeyIndex, endKeyIndex):
        if state.interrupted:
            break
        target = videoFrames[i]
        outTmpFile = os.path.join(outPath, str(keyIndex) + '_' + str(i) + '.png')
        runEb(ebsynth, os.path.join(cwd, sourceFile), os.path.join(cwd, guide), os.path.join(cwd, target),
              os.path.join(cwd, outTmpFile))

    return f'{beginKeyIndex}-{endKeyIndex}:完成'

#改为关键帧引导
def transWithEb(keyIndexs, keyFrames, videoFrames, outPath):
    ebsynth=os.path.join(os.getcwd(),'extensions','sd-webui-video2anime','bin','ebsynth.exe')
    cwd = os.getcwd()
    taskOptions = []
    for idx, keyIndex in enumerate(keyIndexs):
        if state.interrupted:
            break
        sourceFile = keyFrames[idx]
        guide = videoFrames[keyIndex]

        beginKeyIndex = 0
        endKeyIndex = len(videoFrames)

        #最左侧
        if idx-1 < 0:
            beginKeyIndex = 0
        else:
            beginKeyIndex = keyIndexs[idx-1]
        if idx+1 >= len(keyIndexs):
            endKeyIndex = len(videoFrames)
        else:
            endKeyIndex = keyIndexs[idx+1]

        taskOpt = [videoFrames, outPath, keyIndex, ebsynth, cwd, sourceFile, guide, beginKeyIndex, endKeyIndex]
        taskOptions.append(taskOpt)


        # for i in range(beginKeyIndex, endKeyIndex):
        #     if state.interrupted:
        #         break
        #     target = videoFrames[i]
        #     outTmpFile = os.path.join(outPath,str(keyIndex)+'_'+str(i)+'.png')
        #     runEb(ebsynth, os.path.join(cwd, sourceFile), os.path.join(cwd, guide), os.path.join(cwd, target),
        #           os.path.join(cwd, outTmpFile))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(ebTask, tuple(taskOptions))
        print('results:', results)


def giveMeMaskImg(originImg, transImg, maskImg):
    originsingleimg = cv2.bitwise_and(originImg, originImg, mask=maskImg)
    transImg = cv2.bitwise_and(transImg, transImg, mask=maskImg)
    img = cv2.subtract(originImg, originsingleimg)
    img = cv2.add(img, transImg)
    return img

# 将输出的eb帧合并成视频需要的帧
def ebFilesToOutFrames(videoFrames, keyIndexs, outImgPath, outPath, maskDir, isNotNormal):
    outFrames = []
    originBgFrames = []
    greenBgFrames = []
    cwd = os.getcwd()
    for idx, keyIndex in enumerate(keyIndexs):
        beginKeyIndex = keyIndex
        endKeyIndex = len(videoFrames)
        singleRender = False

        if idx + 1 >= len(keyIndexs):
            # 需要单独渲染当前关键帧的图片
            endKeyIndex = len(videoFrames)
            print('单独渲染当前关键帧产生的图片')
            singleRender = True
        else:
            endKeyIndex = keyIndexs[idx + 1]

        for i in range(beginKeyIndex, endKeyIndex):
            if state.interrupted:
                break

            if singleRender:
                currentImg = os.path.join(outImgPath, str(beginKeyIndex) + '_' + str(i) + '.png')
                img = cv2.imread(currentImg, 1)
            else:
                currentImg = os.path.join(outImgPath, str(beginKeyIndex) + '_' + str(i) + '.png')
                nextImg = os.path.join(outImgPath, str(endKeyIndex)+ '_' + str(i) + '.png')
                img_f = cv2.imread(currentImg, 1)
                img_b = cv2.imread(nextImg, 1)
                back_rate = (i - beginKeyIndex) / max(1, (
                            endKeyIndex - beginKeyIndex))
                img = cv2.addWeighted(img_f, 1.0 - back_rate, img_b, back_rate, 0)

            if isNotNormal:
                # 增加原背景输出
                originBgFrame = os.path.join(cwd, outPath, 'origin_' + str(i) + '.png')
                originImg = cv2.imread(videoFrames[i],1)
                maskPath = os.path.join(cwd,maskDir,str(i)+'.png')
                maskImg = cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE)
                originImg = giveMeMaskImg(originImg,img,maskImg)

                # 增加绿色背景输出
                h = img.shape[0]
                w = img.shape[1]
                img_green = np.zeros([h,w,3], np.uint8)
                img_green[:,:,1] = np.zeros([h,w])+255
                greenBgFrame = os.path.join(cwd, outPath, 'green_' + str(i) + '.png')
                greenImg = giveMeMaskImg(img_green,img,maskImg)
                cv2.imwrite(originBgFrame, originImg)
                cv2.imwrite(greenBgFrame, greenImg)
                originBgFrames.append(originBgFrame)
                greenBgFrames.append(greenBgFrame)

            outFrame = os.path.join(cwd, outPath, str(i) + '.png')
            cv2.imwrite(outFrame, img)
            outFrames.append(outFrame)


    return outFrames, originBgFrames, greenBgFrames



def create_white_img(w, h):
    image = np.zeros([h, w, 3], dtype=np.uint8)
    image.fill(255)
    return image


# p 图生图或者文生图实例
# m_file 要转换的视频文件
# fps_scale_child 跳帧参数
# fps_scale_parent 跳帧参数
# max_frames 最大转换帧数 -1代表全转
# m2a_mode 图生图 或者 文生图
# rembg_mode 去背景方式 normal rembg mask
# invoke_tagger 开启反推提示词
# invoke_tagger_val 反推提示词阈值
# common_invoke_tagger 公共提示词
def process_m2a(p, m_file, fps_scale_child, fps_scale_parent, max_frames, m2a_mode, rembg_mode, invoke_tagger, invoke_tagger_val, common_invoke_tagger):
    print('mfile:', m_file)
    images, movie_frames = get_movie_all_images(m_file, fps_scale_child, fps_scale_parent)
    if not images:
        print('Failed to parse the video, please check')
        return
    print(f'The video conversion is completed, images:{len(images)}')
    if max_frames == -1 or max_frames > len(images):
        max_frames = len(images)
    max_frames = int(max_frames)

    p.do_not_save_grid = True
    state.job_count = max_frames

    generate_images = []
    if invoke_tagger:
        refresh_interrogators()

    for i, image in enumerate(images):
        if i >= max_frames:
            break
        state.job = f"{i + 1} out of {max_frames}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')
        img = ImageOps.exif_transpose(img)

        if m2a_mode == 'img2img':
            p.mask = None
            if rembg_mode == 'rembg':
                rembg_img = rembg_mov(img, False)
                mask_img = rembg_mov(img, True)
                img = rembg_img
                p.image_mask = mask_img
            elif rembg_mode == 'maskbg':
                mask_img = rembg_mov(img, True)
                p.image_mask = mask_img
            elif rembg_mode == 'lineart':
                # controlnet 垫图
                p.control_net_input_image = [img,img,img,img]
                img = create_white_img(p.width, p.height) # 全白图片
            # 修改prompt
            if invoke_tagger:
                newTag = getTagsFromImage(img, True, invoke_tagger_val, common_invoke_tagger)
                p.prompt = newTag
                print('p.prompt 改为：', newTag)
            p.init_images = [img] * p.batch_size
            if rembg_mode == 'rembg' or rembg_mode == 'maskbg':
                p.mask_blur = 4
                p.inpainting_fill = 1
                p.inpaint_full_res = False
                p.inpaint_full_res_padding = 32
                p.inpainting_mask_invert = 0

        else:
            # 修改prompt
            if invoke_tagger:
                newTag = getTagsFromImage(img, True, invoke_tagger_val, common_invoke_tagger)
                p.prompt = newTag
                print('p.prompt 改为：', newTag)
            p.init_images = [image]

        print(f'current progress: {i + 1}/{max_frames}')
        processed = process_images(p)
        # 只取第一张

        gen_image = processed.images[0]

        print('这是第',i,'张图片:', gen_image)

        generate_images.append(gen_image)

    if not os.path.exists(m2a_output_dir):
        os.makedirs(m2a_output_dir, exist_ok=True)

    r_f = '.mp4'
    mode = 'mp4v'

    w = p.width
    h = p.height
    print('width,height', w, h)

    print(f'Start generating {r_f} file')

    video = images_to_video(generate_images, movie_frames, mode, w, h,
                            os.path.join(m2a_output_dir, str(int(time.time())) + r_f, ))
    print(f'The generation is complete, the directory::{video}')

    return video

def process_m2a_eb(p, m_file, fps_scale_child, fps_scale_parent, max_frames, m2a_mode, rembg_mode, invoke_tagger, invoke_tagger_val, common_invoke_tagger,min_gap,max_gap,max_delta):
    print('eb渲染')
    print('rembg_mode:', rembg_mode)

    if invoke_tagger:
        refresh_interrogators()

    #工作目录生成
    # work-${time}
    #   key video out
    workDir,keyDir,videoDir,outDir, outTmpDir, maskDir = mkWorkDir()
    print('workDir:', workDir)

    isNotNormal = rembg_mode != 'normal' and rembg_mode != 'lineart'

    # 分拆视频帧到video
    [videoImages, imgDataList, fps] = video2imgs(m_file, videoDir, max_frames, True, p.width, p.height, maskDir, isNotNormal)
    # 挑选关键帧
    [keyImages, keyIndexs] = selectVideoKeyFrame(videoImages,min_gap, max_gap,max_delta, True)

    print('keyIndexs', keyIndexs)

    state.job_count = len(keyIndexs)

    # 将关键帧进行sd转换 生成的图片保存至key目录
    generate_keyFrames = []
    for i, keyIndex in enumerate(keyIndexs):
        image = imgDataList[keyIndex]
        state.job = f"{i + 1} out of {len(keyIndexs)}"
        if state.skipped:
            state.skipped = False
        if state.interrupted:
            break
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')
        img = ImageOps.exif_transpose(img)

        if m2a_mode == 'img2img':
            # p.mask = None
            # if rembg_mode == 'rembg':
            #     rembg_img = rembg_mov(img, False)
            #     mask_img = rembg_mov(img, True)
            #     img = rembg_img
            #     p.image_mask = mask_img
            # elif rembg_mode == 'maskbg':
            #     mask_img = rembg_mov(img, True)
            #     p.image_mask = mask_img
            # 修改prompt
            if invoke_tagger:
                newTag = getTagsFromImage(img, True, invoke_tagger_val, common_invoke_tagger)
                p.prompt = newTag
                print('p.prompt 改为：', newTag)
            if rembg_mode == 'lineart':
                print('设置白色图片')
                # controlnet 垫图
                p.control_net_input_image = [img] * 4
                img = create_white_img(p.width, p.height) # 全白图片
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'RGB')
                img = ImageOps.exif_transpose(img)

            p.init_images = [img] * p.batch_size
            # if rembg_mode == 'rembg' or rembg_mode == 'maskbg':
            #     p.mask_blur = 4
            #     p.inpainting_fill = 1
            #     p.inpaint_full_res = False
            #     p.inpaint_full_res_padding = 32
            #     p.inpainting_mask_invert = 0

        else:
            # 修改prompt
            if invoke_tagger:
                newTag = getTagsFromImage(img, True, invoke_tagger_val, common_invoke_tagger)
                p.prompt = newTag
                print('p.prompt 改为：', newTag)
            p.init_images = [image]

        print(f'current progress: {i + 1}/{len(keyIndexs)}')
        processed = process_images(p)
        # 只取第一张
        gen_image = processed.images[0]
        print('这是第',i,'张关键帧图片:', gen_image)
        keyFramePath = images.save_image(gen_image, keyDir, "", p.seed, p.prompt,
                          forced_filename=str(keyIndexs[i]))
        print('keyFramePath:', keyFramePath[0])
        generate_keyFrames.append(keyFramePath[0])

    # 使用eb进行全video的转换
    # 假设关键帧是1 6 11 16
    # 用关键帧1做source video1做guide video2做target 得out2
    # 切换out2做source video2做guide video3做target 得out3
    # 出到out5的时候 因为out6是关键帧 所以使用key6作为out6
    # 然后用out6 video6 video7 出out7 依次类推

    # 上面效果不好，换成全部由关键帧来引导

    transWithEb(keyIndexs, generate_keyFrames, videoImages, outTmpDir)

    outFrames, originBgFrames, greenBgFrames = ebFilesToOutFrames(videoImages, keyIndexs, outTmpDir, outDir, maskDir,isNotNormal)

    # 将out组装成视频
    r_f = '.mp4'
    mode = 'MP4V'

    w = p.width
    h = p.height
    print('width,height', w, h)
    print(f'Start generating {r_f} file')
    if state.interrupted:
        return
    video = imageFiles_to_video(outFrames, fps, mode, w, h,
                            os.path.join(m2a_output_dir, str(int(time.time())) + r_f, ))

    imageFiles_to_video(originBgFrames, fps, mode, w, h,
                        os.path.join(m2a_output_dir, 'origin_'+str(int(time.time())) + r_f, ))
    imageFiles_to_video(greenBgFrames, fps, mode, w, h,
                        os.path.join(m2a_output_dir, 'green_'+str(int(time.time())) + r_f, ))


    return video
