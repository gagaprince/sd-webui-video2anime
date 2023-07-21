from tagger.tagger import utils
import os.path
import time

import cv2
import numpy
from PIL import Image, ImageOps

import subprocess

import modules.images
import os
from modules.shared import opts, state
from modules import shared, sd_samplers, processing, images
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, process_images, Processed
import rembg
from scripts.m2a_config import m2a_outpath_samples, m2a_output_dir, m2a_eb_output_dir

import time

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
        if tagV > invoke_tagger_val:
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
        img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        video.write(img)
    video.release()
    return out_path

def mkWorkDir():
    if not os.path.exists(m2a_eb_output_dir):
        os.makedirs(m2a_eb_output_dir, exist_ok=True)
    now = time.time()
    now = int(now)
    workDir = os.path.join(m2a_eb_output_dir, str(now))
    keyDir = os.path.join(workDir,'key')
    videoDir = os.path.join(workDir,'video')
    outDir = os.path.join(workDir, 'out')
    os.makedirs(workDir, exist_ok=True)
    return [workDir,keyDir,videoDir,outDir]

def video2imgs(videoPath, imgPath, max_frames):
    if not os.path.exists(imgPath):
        os.makedirs(imgPath, exist_ok=True)             # 目标文件夹不存在，则创建
    cap = cv2.VideoCapture(videoPath)    # 获取视频
    judge = cap.isOpened()                 # 判断是否能打开成功
    fps = cap.get(cv2.CAP_PROP_FPS)      # 帧率，视频每秒展示多少张图片
    count = 0                            # 用于统计保存的图片数量
    imgPaths = []

    while(judge):
        flag, frame = cap.read()         # 读取每一张图片 flag表示是否读取成功，frame是图片
        if not flag:
            print("Process finished!")
            break
        else:
            imgname = str(count)+".png"
            onePath = os.path.join(imgPath,imgname)
            print(onePath)
            cv2.imwrite(onePath, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            imgPaths.append(onePath)
            count += 1
        if max_frames != -1 and count >= max_frames:
            break
    cap.release()

    return [imgPaths, fps]

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
    args = ['cmd','/c',ebsynth, "-style", keyframe, "-guide", guide, frame, "-output", outPath]

    # args = ['cmd','/c','ls']

    print(' '.join(args))

    out = subprocess.run(args, stderr=subprocess.PIPE)

    # If the command fails or verbose level > 1, print the command output
    if out.returncode != 0:
        print("Ebsynth returned a nonzero exit code:")

def transWithEb(keyIndexs, keyFrames, videoFrames, outPath):
    ebsynth=os.path.join(os.getcwd(),'bin','ebsynth.exe')
    videoImageIdx = 0
    keyIdx = 0 #keyIndexs的下标
    keyIndex = keyIndexs[keyIdx] # 关键帧在video中的下标

    sourceFile = ''
    guide = ''
    target = ''
    outTmpFile = ''
    outFrames = []
    for videoImg in videoFrames:
        if videoImageIdx == keyIndex:
            sourceFile = keyFrames[keyIdx]
            guide = videoImg
            target = videoImg
            keyIdx += 1
            if keyIdx >= len(keyIndexs): # 没有关键帧了给一个超大的值
                keyIndex = 999999
            else:
                keyIndex = keyIndexs[keyIdx]
        elif videoImageIdx < keyIndex-1:
            sourceFile = outTmpFile
            guide = videoImg
            target = videoFrames[videoImageIdx+1]
        elif videoImageIdx == keyIndex-1:
            continue
        outTmpFile = os.path.join(outPath,str(videoImageIdx)+'.png')
        runEb(ebsynth, sourceFile, guide, target, outTmpFile)
        videoImageIdx += 1
        outFrames.append(outTmpFile)

    return outFrames


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

def process_m2a_eb(p, m_file, fps_scale_child, fps_scale_parent, max_frames, m2a_mode, rembg_mode, invoke_tagger, invoke_tagger_val, common_invoke_tagger):
    print('eb渲染')

    #工作目录生成
    # work-${time}
    #   key video out
    [workDir,keyDir,videoDir,outDir] = mkWorkDir()
    print('workDir:', workDir)

    # 分拆视频帧到video
    [videoImages, fps] = video2imgs(m_file, videoDir, max_frames)
    # 挑选关键帧
    [keyImages, keyIndexs] = selectKeyFrames(videoImages, fps_scale_parent)

    # 将关键帧进行sd转换 生成的图片保存至key目录
    generate_keyFrames = []
    for i, image in enumerate(keyImages):
        state.job = f"{i + 1} out of {len(videoImages)}"
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

        print(f'current progress: {i + 1}/{max_frames}')
        processed = process_images(p)
        # 只取第一张
        gen_image = processed.images[0]
        print('这是第',i,'张关键帧图片:', gen_image)
        keyFramePath = images.save_image(gen_image, keyDir, "", p.seed, p.prompt,
                          forced_filename=str(keyIndexs[i]))
        generate_keyFrames.append(keyFramePath)

    # 使用eb进行全video的转换
    # 假设关键帧是1 6 11 16
    # 用关键帧1做source video1做guide video2做target 得out2
    # 切换out2做source video2做guide video3做target 得out3
    # 出到out5的时候 因为out6是关键帧 所以使用key6作为out6
    # 然后用out6 video6 video7 出out7 依次类推

    outFrames = transWithEb(keyIndexs, generate_keyFrames, keyImages, outDir)

    # 将out组装成视频
    r_f = '.mp4'
    mode = 'mp4v'

    w = p.width
    h = p.height
    print('width,height', w, h)
    print(f'Start generating {r_f} file')
    video = images_to_video(outFrames, fps, mode, w, h,
                            os.path.join(m2a_output_dir, str(int(time.time())) + r_f, ))

    return video
