import argparse
import json
import os
import tempfile
import time

import numpy as np
import torch

from Modules.Utils import get_ref_idx, write_uintx, write_bytes, read_uintx, read_bytes, calculate_decompression_order
from Network import Network

torch.backends.cudnn.deterministic = True
NUM_LOSSLESS_BITSTREAMS = 4


def lossy_compress(lossy_encoder_path: str, lossy_cfg_path: list, yuv_filepath: str, bin_filepath: str, rec_filepath: str,
                   out_txt_filepath: str, num_slices: int, height: int, width: int, qp: int) -> None:
    """
    Lossy compress slices by VVC.
    """
    enc_cfg = {
        'InputFile': yuv_filepath,
        'BitstreamFile': bin_filepath,
        'ReconFile': rec_filepath,
        'SourceWidth': width,
        'SourceHeight': height,
        'OutputBitDepth': 8,
        'InputChromaFormat': 400,
        'FrameRate': 30,
        'FramesToBeEncoded': num_slices,
        'Level': 6.1,
        'QP': qp,
    }
    enc_cmd = lossy_encoder_path
    for cfg_path in lossy_cfg_path:
        enc_cmd += f' -c {cfg_path}'
    for key, value in enc_cfg.items():
        enc_cmd += f' --{str(key)}={str(value)}'
    enc_cmd += f' > {out_txt_filepath}'
    os.system(enc_cmd)


def lossy_decompress(lossy_decoder_path: str, bin_filepath: str, rec_filepath: str, out_txt_filepath: str) -> None:
    """
    Decompress slices by VVC.
    """
    dec_cmd = f'{lossy_decoder_path} -b {bin_filepath} -o {rec_filepath} -d 8 > {out_txt_filepath}'
    os.system(dec_cmd)


@torch.no_grad()
def lossless_compress(net: Network, ori_slices: list, rec_slices: list) -> list:
    """
    Lossless compress residues by the BCM-Net.
    """
    assert len(ori_slices) == len(rec_slices)
    num_slices = len(ori_slices)
    bitstreams = []
    for slice_idx in range(num_slices):
        ori_slice = torch.from_numpy(ori_slices[slice_idx]).unsqueeze(dim=0).unsqueeze(dim=0).to(next(net.parameters()).device) * 1.
        rec_slice = torch.from_numpy(rec_slices[slice_idx]).unsqueeze(dim=0).unsqueeze(dim=0).to(next(net.parameters()).device) * 1.

        forward_ref_idx, backward_ref_idx = get_ref_idx(slice_idx=slice_idx, num_slices=num_slices)
        ref_forward = None if forward_ref_idx == -1 else \
            torch.from_numpy(ori_slices[forward_ref_idx]).unsqueeze(dim=0).unsqueeze(dim=0).to(next(net.parameters()).device) * 1.
        ref_backward = None if backward_ref_idx == -1 else \
            torch.from_numpy(ori_slices[backward_ref_idx]).unsqueeze(dim=0).unsqueeze(dim=0).to(next(net.parameters()).device) * 1.

        residues = ori_slice - rec_slice
        bitstreams.append(net.compress(residues=residues, x_min=-255, x_max=255, x_tilde=rec_slice, ref_forward=ref_forward, ref_backward=ref_backward))

    return bitstreams


@torch.no_grad()
def lossless_decompress(net: Network, bitstreams: list, rec_slices: list) -> np.ndarray:
    """
    Decompress residues by the BCM-Net.
    """
    assert len(bitstreams) == len(rec_slices)
    num_slices = len(bitstreams)
    decoded_slices = [torch.zeros(1), ] * num_slices
    slices_indices = calculate_decompression_order(num_slices)
    for slice_idx in slices_indices:
        rec_slice = torch.from_numpy(rec_slices[slice_idx]).unsqueeze(dim=0).unsqueeze(dim=0).to(next(net.parameters()).device) * 1.

        forward_ref_idx, backward_ref_idx = get_ref_idx(slice_idx=slice_idx, num_slices=num_slices)
        ref_forward = None if forward_ref_idx == -1 else decoded_slices[forward_ref_idx]

        ref_backward = None if backward_ref_idx == -1 else decoded_slices[backward_ref_idx]
        residues = net.decompress(strings=bitstreams[slice_idx], x_min=-255, x_max=255, x_tilde=rec_slice, ref_forward=ref_forward, ref_backward=ref_backward)

        decoded_slices[slice_idx] = residues + rec_slice

    return torch.stack(decoded_slices, dim=0).squeeze().cpu().numpy().astype(np.uint8)


def merge_bitstreams(lossless_bitstreams: list, lossy_bin_filepath: str, dst_bin_filepath: str, num_slices: int, height: int, width: int):
    """
    Merge head information, lossy and lossless bitstreams and write header info.
    """
    with open(dst_bin_filepath, mode='wb') as f:
        # write header information
        write_uintx(f, value=num_slices, x=16)  # 2 bytes
        write_uintx(f, value=width, x=16)  # 2 bytes
        write_uintx(f, value=height, x=16)  # 2 bytes
        # write length of lossy bitstream
        len_lossy_bitstream = os.path.getsize(lossy_bin_filepath)
        write_uintx(f, value=len_lossy_bitstream, x=32)  # 4 bytes
        # write lossy bitstream to destine bitstream
        with open(lossy_bin_filepath, mode='rb') as src_f:
            data = src_f.read()
            f.write(data)
        # write lossless bitstreams to destine bitstream
        for bitstream in lossless_bitstreams:
            for i in range(NUM_LOSSLESS_BITSTREAMS):
                length = len(bitstream[i])
                write_uintx(f, value=length, x=32)  # 4 bytes
                write_bytes(f, values=bitstream[i])


def parse_bitstreams(bitstream_filepath: str, lossy_bin_filepath: str) -> tuple:
    """
    Parse bitstreams and split to lossy and lossless bitstreams.
    """
    with open(bitstream_filepath, mode='rb') as f:
        # parse header information
        num_slices = read_uintx(f, x=16)
        width = read_uintx(f, x=16)
        height = read_uintx(f, x=16)
        # parse bitstreams of lossy compression
        len_lossy_bitstream = read_uintx(f, x=32)
        with open(lossy_bin_filepath, mode='wb') as lossy_f:
            data = f.read(len_lossy_bitstream)
            lossy_f.write(data)
        # parser bitstreams of lossless compression
        lossless_bitstreams = []
        for i in range(num_slices):
            bitstream = []
            for j in range(NUM_LOSSLESS_BITSTREAMS):
                length = read_uintx(f, x=32)
                bitstream.append(read_bytes(f, n=length))
            lossless_bitstreams.append(bitstream)
    return lossless_bitstreams, num_slices, height, width


@torch.no_grad()
def compress(npy_filepath: str, bin_filepath: str, lossless_net: Network,
             lossy_cfg_path: list, lossy_encoder_path: str, lossy_decoder_path: str, qp: int) -> tuple:
    """
    Compress npy file into bitstreams.
    """
    torch.cuda.synchronize()
    lossy_start_time = time.time()
    assert os.path.splitext(npy_filepath)[-1] == '.npy'
    npy_filename = os.path.split(npy_filepath)[-1]
    # load npy file and convert to slices
    data = np.load(npy_filepath)
    num_slices, height, width = data.shape
    slices = [data[i, :, :] for i in range(num_slices)]
    with tempfile.TemporaryDirectory() as tmp_dir:
        # write to yuv
        yuv_filepath = os.path.join(tmp_dir, os.path.splitext(npy_filename)[0] + '.yuv')
        with open(yuv_filepath, mode='w') as f:
            for s in slices:
                np.asarray(s, dtype=np.uint8).tofile(f)
        # prepare lossy compression configuration
        lossy_bin_filepath = os.path.join(tmp_dir, os.path.splitext(npy_filename)[0] + '.bin')
        lossy_rec_filepath = os.path.join(tmp_dir, os.path.splitext(npy_filename)[0] + '_rec.yuv')
        lossy_out_txt_filepath = os.path.join(tmp_dir, 'out.txt')
        # lossy compression
        lossy_compress(yuv_filepath=yuv_filepath, bin_filepath=lossy_bin_filepath, rec_filepath=lossy_rec_filepath,
                       lossy_cfg_path=lossy_cfg_path, lossy_encoder_path=lossy_encoder_path,
                       out_txt_filepath=lossy_out_txt_filepath, num_slices=num_slices, height=height, width=width, qp=qp)

        # lossy decompression
        lossy_decompress(bin_filepath=lossy_bin_filepath, rec_filepath=lossy_rec_filepath, lossy_decoder_path=lossy_decoder_path, out_txt_filepath=lossy_out_txt_filepath)
        torch.cuda.synchronize()
        lossy_end_time = time.time()
        lossy_runtime = lossy_end_time - lossy_start_time  # runtime in seconds

        # lossless compression of residues
        torch.cuda.synchronize()
        lossless_start_time = time.time()
        with open(yuv_filepath, mode='rb') as f:
            ori_slices = [np.reshape(np.frombuffer(f.read(height * width), 'B'), (height, width)).astype(np.uint8) for _ in range(num_slices)]
        with open(lossy_rec_filepath, mode='rb') as f:
            rec_slices = [np.reshape(np.frombuffer(f.read(height * width), 'B'), (height, width)).astype(np.uint8) for _ in range(num_slices)]
        lossless_bitstreams = lossless_compress(net=lossless_net, ori_slices=ori_slices, rec_slices=rec_slices)

        # merge lossy and lossless bitstreams and write header info
        merge_bitstreams(lossy_bin_filepath=lossy_bin_filepath, dst_bin_filepath=bin_filepath, lossless_bitstreams=lossless_bitstreams,
                         num_slices=num_slices, height=height, width=width)
        bpp = os.path.getsize(bin_filepath) * 8 / num_slices / height / width
        lossy_bpp = os.path.getsize(lossy_bin_filepath) * 8 / num_slices / height / width
        lossless_bpp = bpp - lossy_bpp
        torch.cuda.synchronize()
        lossless_end_time = time.time()
        lossless_runtime = lossless_end_time - lossless_start_time  # runtime in seconds
        return bpp, lossy_bpp, lossless_bpp, lossy_runtime, lossless_runtime


@torch.no_grad()
def decompress(bin_filepath: str, decoded_filepath: str, lossy_decoder_path: str, lossless_net: Network) -> tuple:
    """
    Decompress bitstreams into nii file.
    """
    torch.cuda.synchronize()
    lossy_start_time = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = os.path.split(decoded_filepath)[-1]
        lossy_bin_filepath = os.path.join(tmp_dir, os.path.splitext(filename)[0] + '.bin')

        # parser bitstreams
        lossless_bitstreams, num_slices, height, width = parse_bitstreams(bitstream_filepath=bin_filepath, lossy_bin_filepath=lossy_bin_filepath)

        # lossy decompression
        rec_filepath = os.path.join(tmp_dir, os.path.splitext(filename)[0] + '_rec.yuv')
        lossy_out_txt_filepath = os.path.join(tmp_dir, 'out.txt')
        lossy_decompress(bin_filepath=lossy_bin_filepath, rec_filepath=rec_filepath, lossy_decoder_path=lossy_decoder_path, out_txt_filepath=lossy_out_txt_filepath)
        torch.cuda.synchronize()
        lossy_end_time = time.time()
        lossy_runtime = lossy_end_time - lossy_start_time  # runtime in seconds

        # lossless decompression of residues and add to decoded slices
        torch.cuda.synchronize()
        lossless_start_time = time.time()
        with open(rec_filepath, mode='rb') as f:
            rec_slices = [np.reshape(np.frombuffer(f.read(height * width), 'B'), (height, width)).astype(np.uint8) for _ in range(num_slices)]
        decoded_slices = lossless_decompress(net=lossless_net, bitstreams=lossless_bitstreams, rec_slices=rec_slices)

        # save decoded slices to npy file
        np.save(decoded_filepath, decoded_slices)
        torch.cuda.synchronize()
        lossless_end_time = time.time()
        lossless_runtime = lossless_end_time - lossless_start_time  # runtime in seconds

        return lossy_runtime, lossless_runtime


class Tester:
    def __init__(self) -> None:
        self.args = self.parse_args()
        self.net = Network(bit_depth=self.args.bit_depth).to('cuda' if self.args.gpu else 'cpu').eval()
        self.load_weights()
        self.mr_net_files = [os.path.join(self.args.mr_net_root, npy_filename)
                             for npy_filename in os.listdir(self.args.mr_net_root) if os.path.splitext(npy_filename)[-1] == '.npy']

    @torch.no_grad()
    def test(self) -> None:
        os.makedirs(self.args.save_directory, exist_ok=True)
        record = {}
        for npy_filepath in self.mr_net_files:
            bin_filepath = os.path.join(self.args.save_directory, os.path.splitext(os.path.split(npy_filepath)[-1])[0] + '.bin')
            decoded_filepath = os.path.join(self.args.save_directory, os.path.split(npy_filepath)[-1])

            # compression
            bpp, lossy_bpp, lossless_bpp, lossy_enc_runtime, lossless_enc_runtime = \
                compress(npy_filepath=npy_filepath, bin_filepath=bin_filepath, lossless_net=self.net, qp=self.args.qp,
                         lossy_cfg_path=self.args.lossy_cfg_path, lossy_encoder_path=self.args.lossy_encoder_path, lossy_decoder_path=self.args.lossy_decoder_path)

            # decompression
            lossy_dec_runtime, lossless_dec_runtime = \
                decompress(bin_filepath=bin_filepath, decoded_filepath=decoded_filepath, lossy_decoder_path=self.args.lossy_decoder_path, lossless_net=self.net)

            # examine if decoded slices is identical to original slices
            ori_slices = np.load(npy_filepath)
            decoded_slices = np.load(decoded_filepath)
            assert np.allclose(ori_slices, decoded_slices), f'{decoded_filepath} is not identical to {npy_filepath}'

            # record
            record[npy_filepath] = {
                'bpp': bpp, 'lossy_bpp': lossy_bpp, 'lossless_bpp': lossless_bpp,
                'lossy_enc_runtime': lossy_enc_runtime, 'lossless_enc_runtime': lossless_enc_runtime,
                'lossy_dec_runtime': lossy_dec_runtime, 'lossless_dec_runtime': lossless_dec_runtime
            }

            print(
                npy_filepath,
                ' | bpp = {}, lossy bpp = {}, lossless bpp = {}, '
                'lossy encoding time = {}s, lossless encoding time = {}s, '
                'lossy decoding time = {}s, lossless decoding time = {}s'.format(
                    bpp, lossy_bpp, lossless_bpp, lossy_enc_runtime, lossless_enc_runtime, lossy_dec_runtime, lossless_dec_runtime)
            )

        bpp_average = sum([record[k]['bpp'] for k in record.keys()]) / len(self.mr_net_files)
        lossy_bpp_average = sum([record[k]['lossy_bpp'] for k in record.keys()]) / len(self.mr_net_files)
        lossless_bpp_average = sum([record[k]['lossless_bpp'] for k in record.keys()]) / len(self.mr_net_files)
        lossy_enc_runtime_avg = sum([record[k]['lossy_enc_runtime'] for k in record.keys()]) / len(self.mr_net_files)
        lossless_enc_runtime_avg = sum([record[k]['lossless_enc_runtime'] for k in record.keys()]) / len(self.mr_net_files)
        lossy_dec_runtime_avg = sum([record[k]['lossy_dec_runtime'] for k in record.keys()]) / len(self.mr_net_files)
        lossless_dec_runtime_avg = sum([record[k]['lossless_dec_runtime'] for k in record.keys()]) / len(self.mr_net_files)

        record['average'] = {
            'bpp': bpp_average, 'lossy_bpp': lossy_bpp_average, 'lossless_bpp': lossless_bpp_average,
            'lossy_enc_runtime': lossy_enc_runtime_avg, 'lossless_enc_runtime': lossless_enc_runtime_avg,
            'lossy_dec_runtime': lossy_dec_runtime_avg, 'lossless_dec_runtime': lossless_dec_runtime_avg
        }

        print(
            'Average | bpp = {}, lossy bpp = {}, lossless bpp = {}, '
            'lossy encoding time = {}s, lossless encoding time = {}s, '
            'lossy decoding time = {}s, lossless decoding time = {}s'.format(
                bpp_average, lossy_bpp_average, lossless_bpp_average,
                lossy_enc_runtime_avg, lossless_enc_runtime_avg, lossy_dec_runtime_avg, lossless_dec_runtime_avg
            )
        )

        with open(os.path.join(self.args.save_directory, 'results.json'), mode='w') as f:
            json.dump(record, f)

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--mr_net_root', type=str, help='path of folder of MRNet dataset')

        parser.add_argument('--save_directory', type=str, help='path of folder to save experimental data')

        parser.add_argument('--lossy_encoder_path', type=str, help='path of the lossy encoder')
        parser.add_argument('--lossy_decoder_path', type=str, help='path of the lossy decoder')
        parser.add_argument('--lossy_cfg_path', type=str, nargs='+', help='path of the lossy encoding configuration')
        parser.add_argument('--qp', type=int, default=22, help='QP for lossy encoder')

        parser.add_argument('--checkpoints', type=str, default=None, help='path of checkpoints')
        parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or cpu')

        parser.add_argument('--bit_depth', type=int, default=8, help='bit depth of input data')

        args = parser.parse_args()

        return args

    def load_weights(self) -> None:
        print(f'\n===========Load model_weights from {self.args.checkpoints}===========\n')
        ckpt = torch.load(self.args.checkpoints, map_location='cuda' if self.args.gpu else 'cpu')
        self.net.load_state_dict(ckpt['network'] if 'network' in ckpt else ckpt)


if __name__ == '__main__':
    tester = Tester()
    tester.test()
