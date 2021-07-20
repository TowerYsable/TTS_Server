import argparse
import os
import subprocess

import numpy as np
import torch
import yaml
from scipy.io import wavfile
import wave

from mtts.models.fs2_model import FastSpeech2
from mtts.models.vocoder import *
from mtts.text import TextProcessor
from mtts.utils.logging import get_logger
from mtts.text.gp2py import TextNormal

logger = get_logger(__file__)

def check_ffmpeg():
    r, path = subprocess.getstatusoutput("which ffmpeg")
    return r == 0


with_ffmpeg = check_ffmpeg()


def build_vocoder(device, config):
    vocoder_name = config['vocoder']['type']
    VocoderClass = eval(vocoder_name)
    model = VocoderClass(**config['vocoder'][vocoder_name])
    return model


def normalize(wav):
    assert wav.dtype == np.float32
    eps = 1e-6
    sil = wav[1500:2000]
    #wav = wav - np.mean(sil)
    #wav = (wav - np.min(wav))/(np.max(wav)-np.min(wav)+eps)
    wav = wav / np.max(np.abs(wav))
    #wav = wav*2-1
    wav = wav * 32767
    return wav.astype('int16')


def to_int16(wav):
    wav = wav = wav * 32767
    wav = np.clamp(wav, -32767, 32768)
    return wav.astype('int16')

with open('./checkmodel/config.yaml') as f:
    config = yaml.safe_load(f)
    logger.info(f.read())
model = FastSpeech2(config)
sd = torch.load('./checkmodel/checkpoint_1250000.pth.tar',map_location='cuda')
model.load_state_dict(sd)
model = model.to('cuda')
torch.set_grad_enabled(False)
vocoder = build_vocoder('cuda',config=config)


def syn_server(text):
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    with open('./checkmodel/config.yaml') as f:
        config = yaml.safe_load(f)
        logger.info(f.read())

    sr = config['fbank']['sample_rate']
    text_processor = TextProcessor(config)

    tn = TextNormal('gp.vocab', 'py.vocab', add_sp1=True, fix_er=True)
    py_list, gp_list = tn.gp2py(text)
    print(py_list, gp_list)
    if os.path.exists('input.txt'):
        os.remove('input.txt')
    for py, gp in zip(py_list, gp_list):
        txt_text = py + '|' + gp
        with open('input.txt', 'a') as f:
            f.write(txt_text + '\n')
            print()

    try:
        lines = open('input.txt').read().split('\n')
    except:
        print("failed")
        lines = ['input.txt']

    index = -1
    for line in lines:
        index = index +1
        print("lines",index)
        if len(line) == 0 or line.startswith('#'):
            continue
        logger.info(f'processing {line}')
        name, tokens = text_processor(line)
        tokens = tokens.to('cuda')
        seq_len = torch.tensor([tokens.shape[1]])
        tokens = tokens.unsqueeze(1)
        seq_len = seq_len.to('cuda')
        max_src_len = torch.max(seq_len)
        output = model(tokens, seq_len, max_src_len=max_src_len, d_control=1.0)
        mel_pred, mel_postnet, d_pred, src_mask, mel_mask, mel_len = output

        # convert to waveform using vocoder
        mel_postnet = mel_postnet[0].transpose(0, 1).detach()
        mel_postnet += config['fbank']['mel_mean']
        wav = vocoder(mel_postnet)
        if config['synthesis']['normalize']:
            wav = normalize(wav)
        else:
            wav = to_int16(wav)
        dst_file = os.path.join('./outputs', f'{index+1}.wav')
        print("dst_file",dst_file)
        #np.save(dst_file+'.npy',mel_postnet.cpu().numpy())
        if not os.path.exists(dst_file):
            print("pass")
            wavfile.write(dst_file, sr, wav)
        else:
            print("psss")
            os.remove(dst_file)
            wavfile.write(dst_file, sr, wav)

        logger.info(f'writing file to {dst_file}')

        # TODO:
    if index >1:
        infiles = [f"./outputs/{index-1}.wav",f"./outputs/{index}.wav"]
        output_wav = "./outputs/output.wav"
        data = []
        for infile in infiles:
            w = wave.open(infile,'rb')
            data.append([w.getparams(), w.readframes(w.getnframes())])
            w.close()
        output = wave.open(output_wav, 'wb')
        output.setparams(data[0][0])
        output.writeframes(data[0][1])
        output.writeframes(data[1][1])
        output.close()
        return output_wav
    elif(index==1):
        return f"./outputs/{index}.wav"
        # return dst_file

# def pro_text(text):


# pro_text(text="实验室在正前方,让我带你去吧")
# syn_server(text="实验室在正前方,让我带你去吧")