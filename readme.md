```bash
conda create -y -n lerobot_dagger python=3.12
conda activate lerobot_dagger

conda install ffmpeg 
pip3 install -e .
pip install lerobot[dataset]
pip install lerobot[sarm]
python -m pip install -U qwen-vl-utils
pip3 install matplotlib 
pip install faker
pip install wandb
pip install pyarrow
pip install lerobot[dataset]
pip install lerobot[sarm]
```