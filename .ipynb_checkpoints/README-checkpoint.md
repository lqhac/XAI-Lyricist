# XAILyricist
This is the official repository for the paper "XAI-Lyricist: Improving the Singability of AI-Generated Lyrics with Prosody Explanations".

## Usage

1. build dictiory
```python
python ./0_build_dict/build_dictionary.py -config ./configs/configs.yaml
```

2. binarise data
```python
python ./1_data_binarisation/binarise.py -config ./configs/configs.yaml
```

3. [Test Unit] Dataloader
```python
python ./2_test_dataloader/dataloader.py -config ./configs/configs.yaml
```

4. training
```python
python ./3_train_bart/train.py -config ./configs/configs.yaml
```

5. inference
```python
python ./4_infer_bart/inference.py -config ./configs/configs.yaml
```


### To cite
```
@inproceedings{ijcai2024p872,
  title     = {XAI-Lyricist: Improving the Singability of AI-Generated Lyrics with Prosody Explanations},
  author    = {Liang, Qihao and Ma, Xichu and Doshi-Velez, Finale and Lim, Brian and Wang, Ye},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {7877--7885},
  year      = {2024},
  month     = {8},
  note      = {Human-Centred AI},
  doi       = {10.24963/ijcai.2024/872},
  url       = {https://doi.org/10.24963/ijcai.2024/872},
}
```