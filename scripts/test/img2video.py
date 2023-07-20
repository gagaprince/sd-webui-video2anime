import cv2
import os
def img2video(pic_path, video_name, start, end, fps = 30):
    video_file = os.path.join(pic_path, video_name)
    tmp_file = os.path.join(pic_path, str(start) + '.png')
    print(tmp_file)
    tmpimg = cv2.imread(tmp_file)
    imginfo = tmpimg.shape
    size = (imginfo[1], imginfo[0])
    print('视频size', size)
    codec = cv2.VideoWriter_fourcc(*"MP4V")
    vid = cv2.VideoWriter(video_file, codec, fps, size)

    i = start
    seed = '3459414726'
    while i <= end:
        # if i%25 == 0:
        #     seed += 1
        filename = str(i)+ '.png'
        filename = os.path.join(pic_path, filename)
        img = cv2.imread(filename, 1)
        # 直接写入图片对应的数据
        vid.write(img)
        # vid.write(img)
        #vid.write(img)
        i += 1
    vid.release()

    print('out file is:', video_file)

def main():
    pic_path = 'G:\\webui\\videowork\\genvideo\\20230719\\out1\\'
    videoname = '1.mp4'
    img2video(pic_path, videoname, 1, 149, 30)


if __name__ == '__main__':
    main()
