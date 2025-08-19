# CLI tool for TTS generation and Voice Cloning from CSV file with F5-TTS

[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://vshymanskyy.github.io/StandWithUkraine/)

**Installation:**
> Python 3.12

> pip install -r requirements.txt

If you want to try this code with CUDA enabled, install torch with following command:
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

Or use this page for getting of latest or specified version of torch:  
> https://pytorch.org/get-started/locally/

**How to use**:
> python vc-from-csv.py --csv "/<path_to_repo>/samples.csv"

> python vc-from-csv.py --csv "/<path_to_repo>/samples.csv" --ref-wav "/<path_to_repo>/examples/my-voice-emotioned-clean.wav"

> python vc-from-csv.py --csv "/<path_to_repo>/examples/samples.csv" --ref-wav "/<path_to_repo>/examples/untitled.wav" --ref-txt "some call me nature, others call me mother nature"

- See csv file example > examples/samples.csv

This code uses default F5-TTS model