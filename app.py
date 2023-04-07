import os
import sys
import html
import glob
import uuid
import hashlib
import requests
from tqdm import tqdm

os.system("git clone https://github.com/FrozenBurning/SceneDreamer.git")
os.system("cp -r SceneDreamer/* ./")

pretrained_model = dict(file_url='https://drive.google.com/uc?id=1IFu1vNrgF1EaRqPizyEgN_5Vt7Fyg0Mj',
                            alt_url='', file_size=330571863,
                            file_path='./scenedreamer_released.pt',)


def download_file(session, file_spec, use_alt_url=False, chunk_size=128, num_attempts=10):
    file_path = file_spec['file_path']
    if use_alt_url:
        file_url = file_spec['alt_url']
    else:
        file_url = file_spec['file_url']

    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    progress_bar = tqdm(total=file_spec['file_size'], unit='B', unit_scale=True)
    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        progress_bar.reset()
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)

            # Validate.
            if 'file_size' in file_spec and data_size != file_spec['file_size']:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec['file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            break

        except Exception as e:
            # print(e)
            # Last attempt => raise error.
            if not attempts_left:
                raise

            # Handle Google Drive virus checker nag.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                links = [html.unescape(link) for link in data.decode('utf-8').split('"') if 'confirm=t' in link]
                if len(links) == 1:
                    file_url = requests.compat.urljoin(file_url, links[0])
                    continue

    progress_bar.close()

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass

print('Downloading SceneDreamer pretrained model...')
with requests.Session() as session:
    try:
        download_file(session, pretrained_model)
    except:
        print('Google Drive download failed.\n')



import os
import torch
import torch.nn as nn
import importlib
import argparse
from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
import gradio as gr
from PIL import Image


class WrappedModel(nn.Module):
    r"""Dummy wrapping the module.
    """

    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        r"""PyTorch module forward function overload."""
        return self.module(*args, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', type=str, default='./configs/scenedreamer_inference.yaml', help='Path to the training config file.')
    parser.add_argument('--checkpoint', default='./scenedreamer_released.pt',
                        help='Checkpoint path.')
    parser.add_argument('--output_dir', type=str, default='./test/',
                        help='Location to save the image outputs')
    parser.add_argument('--seed', type=int, default=8888,
                        help='Random seed.')
    args = parser.parse_args()
    return args


args = parse_args()
cfg = Config(args.config)

# Initialize cudnn.
init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

# Initialize data loaders and models.

lib_G = importlib.import_module(cfg.gen.type)
net_G = lib_G.Generator(cfg.gen, cfg.data)
net_G = net_G.to('cuda')
net_G = WrappedModel(net_G)

if args.checkpoint == '':
    raise NotImplementedError("No checkpoint is provided for inference!")

# Load checkpoint.
# trainer.load_checkpoint(cfg, args.checkpoint)
checkpoint = torch.load(args.checkpoint, map_location='cpu')
net_G.load_state_dict(checkpoint['net_G'])

# Do inference.
net_G = net_G.module
net_G.eval()
for name, param in net_G.named_parameters():
    param.requires_grad = False
torch.cuda.empty_cache()
world_dir = os.path.join(args.output_dir)
os.makedirs(world_dir, exist_ok=True)



def get_bev(seed):
    print('[PCGGenerator] Generating BEV scene representation...')
    os.system('python terrain_generator.py --size {} --seed {} --outdir {}'.format(net_G.voxel.sample_size, seed, world_dir))
    heightmap_path = os.path.join(world_dir, 'heightmap.png')
    semantic_path = os.path.join(world_dir, 'colormap.png')
    heightmap = Image.open(heightmap_path)
    semantic = Image.open(semantic_path)
    return semantic, heightmap

def get_video(seed, num_frames):
    device = torch.device('cuda')
    rng_cuda = torch.Generator(device=device)
    rng_cuda = rng_cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    net_G.voxel.next_world(device, world_dir, checkpoint)
    cam_mode = cfg.inference_args.camera_mode
    current_outdir = os.path.join(world_dir, 'camera_{:02d}'.format(cam_mode))
    os.makedirs(current_outdir, exist_ok=True)
    z = torch.empty(1, net_G.style_dims, dtype=torch.float32, device=device)
    z.normal_(generator=rng_cuda)
    net_G.inference_givenstyle(z, current_outdir, **vars(cfg.inference_args))
    return os.path.join(current_outdir, 'rgb_render.mp4')

markdown=f'''
  # SceneDreamer: Unbounded 3D Scene Generation from 2D Image Collections
  
  Authored by Zhaoxi Chen, Guangcong Wang, Ziwei Liu
  ### Useful links:
  - [Official Github Repo](https://github.com/FrozenBurning/SceneDreamer)
  - [Project Page](https://scene-dreamer.github.io/)
  - [arXiv Link](https://arxiv.org/abs/2302.01330)
  Licensed under the S-Lab License.
  First use the button "Generate BEV" to randomly sample a 3D world represented by a height map and a semantic map. Then push the button "Render" to generate a camera trajectory flying through the world.
'''

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown(markdown)
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    semantic = gr.Image(type="pil", shape=(2048, 2048))
                with gr.Column():
                    height = gr.Image(type="pil", shape=(2048, 2048))
            with gr.Row():
                # with gr.Column():
                #     image = gr.Image(type='pil', shape(540, 960))
                with gr.Column():
                    video=gr.Video()
    with gr.Row():
      num_frames = gr.Slider(minimum=10, maximum=200, value=10, label='Number of rendered frames')
      user_seed = gr.Slider(minimum=0, maximum=999999, value=8888, label='Random seed')

    with gr.Row():
        btn = gr.Button(value="Generate BEV")
        btn_2=gr.Button(value="Render")

    btn.click(get_bev,[user_seed],[semantic, height])
    btn_2.click(get_video,[user_seed, num_frames],[video])

demo.launch(debug=True)