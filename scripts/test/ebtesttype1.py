import os
import subprocess
import sys

sys.path.append('E:\\work\\py\\smallvideo\\bin\\')



def runEb(ebsynth, keyframe, guide, frame, outPath):
    args = ['cmd','/c',ebsynth, "-style", keyframe, "-guide", guide, frame, "-output", outPath]

    # args = ['cmd','/c','ls']

    print(' '.join(args))

    out = subprocess.run(args, stderr=subprocess.PIPE)

    # If the command fails or verbose level > 1, print the command output
    if out.returncode != 0:
        print("Ebsynth returned a nonzero exit code:")

def rendFrames(keyPath,videoPath,keyNum,perNum,outputPath):
    ebsynth = 'E:\\work\\py\\smallvideo\\bin\\ebsynth.exe'
    index = 0
    for idx in range(keyNum):
        keyFrame = os.path.join(keyPath,str(idx).rjust(5,'0')+'--((masterpiece)),best quality,ultra-detailed,smooth,best illustrationbest, shadow,photorealistic,hyperrealistic,backlighting,(ani.png')
        print(keyFrame)
        begin = idx * perNum - int(perNum / 2) + 1
        print('key idx:',idx,'begin:',begin)
        keyVideoFram = os.path.join(videoPath,'png_'+str(idx * perNum+1).rjust(4,'0')+'.png')
        print('keyVideoFrame:', keyVideoFram)
        for videoIdx in range(perNum):
            currentVideoFrame = begin + videoIdx
            if currentVideoFrame < 1 or currentVideoFrame > keyNum * perNum:
                continue
            print('currentVideoFrame:',currentVideoFrame)
            videoFrame = os.path.join(videoPath,'png_'+str(currentVideoFrame).rjust(4,'0')+'.png')
            print('videoFrame:', videoFrame)

            outVideoFrame = os.path.join(outputPath,str(index)+'.png')
            index += 1
            print('outVideoFrame:', outVideoFrame)

            runEb(ebsynth, keyFrame, keyVideoFram,videoFrame,outVideoFrame)






def main():
    keyPath = 'G:\\webui\\videowork\\genvideo\\20230719\\keys'
    videoPath = 'G:\\webui\\videowork\\genvideo\\20230719\\video'
    outputPath = 'G:\\webui\\videowork\\genvideo\\20230719\\out1'
    rendFrames(keyPath,videoPath,30,5,outputPath)



if __name__ == '__main__':
    main()
