import argparse
import utils.datasets as datasets
import utils.utils_image as util
import math
import torch
from angular_spectrum_method_pytorch import ASM
from model import PDRNet as create_model
import os
import numpy as np
import time
import cv2
from skimage.metrics import structural_similarity as ssim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

m = 1.
cm = 1e-2
um = 1e-6
mm = 1e-3
nm = 1e-9

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--data_root', type=str, default='./data/test')
    parser.add_argument('--n_channels', type=int, default=3,
                        help='specify the image channel. R:0, G:1, B:2, RGB:3')
    parser.add_argument('--target-size', type=tuple, default=(1160, 2000),
                        help='You can specify the target resolution.(H,W)')
    parser.add_argument('--roi-size', type=tuple, default=(1080, 1920), help='specify the ROI. None or (1080, 1920)')
    parser.add_argument('--linearRGB', type=bool, default=False, help='linearize intensity to linearRGB')
    parser.add_argument('--I2A', type=bool, default=False, help='intensity to amplitude')
    # propagation
    parser.add_argument('--pitch', type=float, default=8 * um, help='the pixel pitch of slm. 6.4um or 8um')
    parser.add_argument('--z', type=float, default=20 * cm, help='light field propagation distance')
    parser.add_argument('--wavelength', type=int, default=532 * nm, help='638 nm, 532 nm, 450 nm')
    parser.add_argument('--band_limited', type=bool, default=True, help='')
    # adjust the light field intensity during propagation
    parser.add_argument('--adjust-intensity', type=bool, default=True,
                        help='adjust intensity, Too high intensity will affect the diffraction results')
    parser.add_argument('--intensity-factor', type=float, default=0.38, help='Specifies the intensity factor')

    parser.add_argument('--weight-dir', type=str, default='./runs/train/532nm_20cm_2022-04-10_19-58-22')
    parser.add_argument('--save-dir', type=str, default='./runs/test')

    opt = parser.parse_args()
    return opt


def test(opt):
    save_dir = os.path.join(opt.save_dir, os.path.split(opt.weight_dir)[-1])
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/logs.txt', 'w') as f:
        f.write('>{}<\n'.format('+'*55))
        f.write(f"weight: '{opt.weight_dir}'\n")
        f.write('>{}<\n'.format('+'*55))

    dataset = datasets.CustomDataSet(data_root=opt.data_root,
                                     n_channels=opt.n_channels,
                                     target_size=opt.target_size,
                                     roi_size=opt.roi_size,
                                     linearRGB=opt.linearRGB,
                                     I2A=opt.I2A)
    print(f" -- {len(dataset)} images in '{opt.data_root}'.")

    prop = ASM(Nx=opt.target_size[1], Ny=opt.target_size[0], pitch=opt.pitch,
               wavelength=opt.wavelength, band_limited=opt.band_limited, device=device)

    model = create_model.PDRNet(opt, prop).to(device)
    weight_path = os.path.join(opt.weight_dir, 'epoch_99.pth')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    model.warmup(device, imgsize=(1, 1, *opt.target_size), half=False)
    for intensity in dataset:
        with torch.no_grad():
            intensity, filename = intensity
            intensity = intensity[None].to(device)

            # intensity to slm phase
            stime = time.time()
            hologram = model(intensity)
            etime = time.time()

            # slm phase to intensity
            recon = util.holo_to_recon(opt, prop, hologram, intensity=None)

            inten_array = (intensity * 255.0).cpu().detach().squeeze().numpy().astype(np.uint8)
            holo_array = (255.0*(hologram+math.pi)/2/math.pi).cpu().detach().squeeze().numpy().astype(np.uint8)
            recon_array = (recon * 255.0).cpu().detach().squeeze().numpy().astype(np.uint8)

            # calculate psnr and ssim
            ssim_val = ssim(inten_array, recon_array)

        info = '< {:12}  ssim:{:6.3f}  time:{:5.3f}s >'.format(filename, ssim_val, etime-stime)   
        with open(f'{save_dir}/logs.txt', 'a') as f:
            f.write(info+'\n')
        print(info)

        idx = filename.split('.')[0]
        cv2.imwrite(os.path.join(save_dir, f'{idx}_inten.bmp'), inten_array)
        cv2.imwrite(os.path.join(save_dir, f'{idx}_holo.bmp'), holo_array)
        cv2.imwrite(os.path.join(save_dir, f'{idx}_recon.bmp'), recon_array)
    print("\nSaved in '{}'".format(save_dir))



if __name__ == '__main__':
    opt = parse_opt(True)
    test(opt)
