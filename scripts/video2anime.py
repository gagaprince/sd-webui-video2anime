import gradio as gr
from pathlib import Path
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules import script_callbacks, shared, scripts
from modules.ui_components import ToolButton, FormRow, FormGroup
from modules.ui_common import folder_symbol, plaintext_to_html
import json
import os
import shutil
import sys
import platform
import subprocess as sp
from modules.shared import opts, state

from scripts.m2a_config import m2a_outpath_samples, m2a_output_dir
# from scripts.xyz import init_xyz

from scripts.m2a_util import process_m2a, process_m2a_eb

NAME = 'movie2anime'

print(NAME)




def open_folder(f):
    print('打开文件夹：', m2a_output_dir)
    if not os.path.exists(f):
        print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
        return
    elif not os.path.isdir(f):
        print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
        return

    if not shared.cmd_opts.hide_ui_dir_config:
        path = os.path.normpath(f)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            sp.Popen(["open", path])
        elif "microsoft-standard-WSL2" in platform.uname().release:
            sp.Popen(["wsl-open", path])
        else:
            sp.Popen(["xdg-open", path])

class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.m2aScriptIsRuning = False

    def title(self):
        return NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(NAME, open=False):
            enabled = gr.Checkbox(label='Enabled', value=False)
            with FormRow().style(equal_height=False):
                with gr.Column(variant='compact', elem_id=f"m2a_settings"):
                    init_mov = gr.Video(label="Video for movie2anime", elem_id="m2a_video", show_label=False,
                                        source="upload")  # .style(height=480)
                    init_mov_dir = gr.Textbox(label="批量转换视频目录", elem_id="m2a_video_dir", show_label=True, lines=1,
                                              placeholder="请输入批量视频目录, 上传视频与当前选项二选一")
            with FormRow():
                transform_mode = gr.Radio(label="转换方式", elem_id="transform_mode",
                                       choices=["多帧渲染", "关键帧渲染"], type="value", value="多帧渲染")
            with FormRow():
                min_gap = gr.Number(label='最小关键帧间隔', value=10,
                                            elem_id='min_gap')
                max_gap = gr.Number(label='最大关键帧间隔', value=300,
                                    elem_id='max_gap')
                max_delta = gr.Number(label='选取关键帧阈值', value=35,
                                    elem_id='max_delta')
            with FormRow():
                invoke_tagger = gr.Checkbox(label='是否启用反推提示词', value=False)
            with FormRow():
                invoke_tagger_val = gr.Number(label='反推提示词阈值', value=0.3,
                                            elem_id='m2a_invoke_tag_val')
            with FormRow():
                common_invoke_tagger = gr.Textbox(label="如果你使用反推提示词，请输入你想附加的正向tag", elem_id="m2a_common_invoke_tagger", show_label=True, lines=3,
                                              placeholder="通用正向提示词，例如：((masterpiece)),best quality,ultra-detailed,smooth,best illustrationbest, shadow,photorealistic,hyperrealistic,backlighting,")
            with FormRow():
                rembg_mode = gr.Radio(label="去背景模式", elem_id="rembg_mode",
                                       choices=["正常", "透明背景", "原视频背景"], type="index", value="0")
            with FormRow():
                fps_scale_child = gr.Number(label='跳帧分子--假设你想m帧内取n帧，此处填n', value=1, elem_id='m2a_fps_scale_child')
                fps_scale_parent = gr.Number(label='跳帧分母--假设你想m帧内取n帧，此处填m', value=1, elem_id='m2a_fps_scale_parent')
            with FormRow():
                max_frames = gr.Number(label='最大转换帧数，-1代表整体转换', value=-1, elem_id='m2a_max_frame')
            with FormRow():
                open_folder_button = gr.Button(folder_symbol,
                                               elem_id="打开生成目录")

            open_folder_button.click(
                fn=lambda: open_folder(m2a_output_dir),
                inputs=[],
                outputs=[],
            )
        return [
            enabled,
            init_mov,
            init_mov_dir,
            rembg_mode,
            fps_scale_child,
            fps_scale_parent,
            invoke_tagger,
            invoke_tagger_val,
            common_invoke_tagger,
            max_frames,
            transform_mode,
            min_gap,
            max_gap,
            max_delta,
        ]

    def multiRender(
            self,
            p: StableDiffusionProcessing,
            init_mov: str,
            init_mov_dir: str,
            rembg_mode: int,
            fps_scale_child: int,
            fps_scale_parent: int,
            invoke_tagger: bool,
            invoke_tagger_val: int,
            common_invoke_tagger: str,
            max_frames: int,
            m2a_mode: str,
    ):
        print('多帧渲染')
        if rembg_mode == 0:
            rembg_mode = 'normal'
        elif rembg_mode == 1:
            rembg_mode = 'rembg'
        else:
            rembg_mode = 'maskbg'

        videos = []

        if not init_mov_dir:
            video = process_m2a(p, init_mov, fps_scale_child, fps_scale_parent, max_frames, m2a_mode, rembg_mode,
                                invoke_tagger, invoke_tagger_val, common_invoke_tagger)
            videos.append(video)
        else:
            m_files = os.listdir(init_mov_dir)
            for file_name in m_files:
                m_file = os.path.join(init_mov_dir, file_name)
                video = process_m2a(p, m_file, fps_scale_child, fps_scale_parent, max_frames, m2a_mode, rembg_mode,
                                    invoke_tagger, invoke_tagger_val, common_invoke_tagger)
                videos.append(video)

        for video in videos:
            print('video complete, output file is ', video)

    def keyFrameRender(
        self,
        p: StableDiffusionProcessing,
        init_mov: str,
        init_mov_dir: str,
        rembg_mode: int,
        fps_scale_child: int,
        fps_scale_parent: int,
        invoke_tagger: bool,
        invoke_tagger_val: int,
        common_invoke_tagger: str,
        max_frames: int,
        m2a_mode: str,
        min_gap: int,
        max_gap: int,
        max_delta: int,
    ):
        videos = []

        if not init_mov_dir:
            video = process_m2a_eb(p, init_mov, fps_scale_child, fps_scale_parent, max_frames, m2a_mode, rembg_mode,
                                invoke_tagger, invoke_tagger_val, common_invoke_tagger,min_gap,max_gap,max_delta)
            videos.append(video)
        else:
            m_files = os.listdir(init_mov_dir)
            for file_name in m_files:
                m_file = os.path.join(init_mov_dir, file_name)
                video = process_m2a_eb(p, m_file, fps_scale_child, fps_scale_parent, max_frames, m2a_mode, rembg_mode,
                                    invoke_tagger, invoke_tagger_val, common_invoke_tagger,min_gap,max_gap,max_delta)
                videos.append(video)

        for video in videos:
            print('video complete, output file is ', video)

    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        init_mov: str,
        init_mov_dir: str,
        rembg_mode: int,
        fps_scale_child: int,
        fps_scale_parent: int,
        invoke_tagger: bool,
        invoke_tagger_val: int,
        common_invoke_tagger: str,
        max_frames: int,
        transform_mode: str,
        min_gap: int,
        max_gap: int,
        max_delta: int,
    ):

        if enabled and not self.m2aScriptIsRuning:
            try:
                self.m2aScriptIsRuning = True
                noise_multiplier = 1
                m2a_mode = 'img2img'
                if isinstance(p, StableDiffusionProcessingTxt2Img):
                    m2a_mode = 'text2img'
                print('movie2anime process:', enabled, init_mov, init_mov_dir, rembg_mode, fps_scale_child, fps_scale_parent,invoke_tagger,invoke_tagger_val,common_invoke_tagger, max_frames)
                print('m2a mode:', m2a_mode)
                if isinstance(p, StableDiffusionProcessingImg2Img):
                    print('movie2anime process resize_mode:', p.resize_mode)

                if not init_mov and not init_mov_dir:
                    raise Exception('Error！ Please add a video file!')

                if transform_mode == '多帧渲染':
                    self.multiRender(
                        p,
                        init_mov,
                        init_mov_dir,
                        rembg_mode,
                        fps_scale_child,
                        fps_scale_parent,
                        invoke_tagger,
                        invoke_tagger_val,
                        common_invoke_tagger,
                        max_frames,
                        m2a_mode,
                    )
                else:
                    self.keyFrameRender(
                        p,
                        init_mov,
                        init_mov_dir,
                        rembg_mode,
                        fps_scale_child,
                        fps_scale_parent,
                        invoke_tagger,
                        invoke_tagger_val,
                        common_invoke_tagger,
                        max_frames,
                        m2a_mode,
                        min_gap,
                        max_gap,
                        max_delta,
                    )

            finally:
                self.m2aScriptIsRuning = False
                state.interrupted = True





# init_xyz(Script, NAME)