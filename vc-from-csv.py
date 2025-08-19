import os
import argparse
import pandas as pd
import torch
from f5_tts.api import F5TTS

"""

python vc-from-csv.py --csv "/home/ailus/Projects/RD-GenAI-Course/GenAi-Course/Lecture-14-VoiceCloning/examples/samples.csv" --ref-wav "/home/ailus/Projects/RD-GenAI-Course/GenAi-Course/Lecture-14-VoiceCloning/examples/my-voice-emotioned-clean.wav"
python vc-from-csv.py --csv "/home/ailus/Projects/RD-GenAI-Course/GenAi-Course/Lecture-14-VoiceCloning/examples/samples.csv" --ref-wav "/home/ailus/Projects/RD-GenAI-Course/GenAi-Course/Lecture-14-VoiceCloning/examples/untitled.wav"

"""

DIR_CURRENT = os.getcwd()

args = argparse.ArgumentParser()
args.add_argument('--csv', type=str, dest='file', required=True, help='Path to csv file to process with column: tts')
args.add_argument('--ref-wav', type=str, dest='ref_wav_file', default=DIR_CURRENT + "/examples/basic/basic_ref_en.wav", required=False, help='Path to wav file to clone')
args.add_argument('--ref-txt', type=str, dest='ref_txt', default="some call me nature, others call me mother nature.", required=False, help='Text transcription of --ref-wav')
args = args.parse_args()

if not os.path.isfile(args.file):
    raise FileNotFoundError(args.file)

DIR_OUTPUTS = DIR_CURRENT + '/outputs/'
if not os.path.isdir(DIR_OUTPUTS):
    os.mkdir(DIR_OUTPUTS)

if not os.path.isfile(args.ref_wav_file):
    raise FileNotFoundError(f"Provided --ref_wav_file {args.ref_wav_file} does not exist")

df = pd.read_csv(args.file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device is: {device}")
f5tts = F5TTS(device=device)

for index, row in df.iterrows():
    f5tts.infer(
        ref_file=args.ref_wav_file,
        ref_text=args.ref_txt,
        gen_text=row['tts'],
        file_wave=str(DIR_OUTPUTS + f"{index}.wav"),
        file_spec=str(DIR_OUTPUTS + f"{index}.png"),
        seed=None,
    )