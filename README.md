# global-wheat-detection

Reviewing Past Kaggle Solutions from Global Wheat Detection comp.

Notebooks:
- starter notebook

To run DDP training in `scripts/src/trainer.py`

- Download dataset from [here](https://www.kaggle.com/competitions/global-wheat-detection/data)
- `pip install -r requirements.txt`
- `torchrun --standalone --nproc_per_node=gpu trainer.py 10 2` 

