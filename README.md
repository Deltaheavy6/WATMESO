WATMESO
modified on Improved DeepFake Detection Using Whisper Features

The paper for the original research is available [here](https://www.isca-speech.org/archive/interspeech_2023/kawa23b_interspeech.html).


## Before you start

### Whisper
To download Whisper encoder used in training run `download_whisper.py`.

### Datasets

Download appropriate datasets:
* [ASVspoof2021 DF subset](https://zenodo.org/record/4835108) (**Please note:** we use this [keys&metadata file](https://www.asvspoof.org/resources/DF-keys-stage-1.tar.gz), directory structure is explained [here](https://github.com/piotrkawa/deepfake-whisper-features/issues/7#issuecomment-1830109945)),
* [In-The-Wild dataset](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild).



### Dependencies
Install required dependencies using (we assume you're using conda and the target env is active):
```bash
bash install.sh
```

List of requirements:
```
python=3.8
pytorch==1.11.0
torchaudio==0.11
asteroid-filterbanks==0.4.0
librosa==0.9.2
openai whisper (git+https://github.com/openai/whisper.git@7858aa9c08d98f75575035ecd6481f462d66ca27)
```

### 
