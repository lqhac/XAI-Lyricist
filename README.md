# XAILyricist
This is the official repository for the paper "XAI-Lyricist: Improving the Singability of AI-Generated Lyrics with Prosody Explanations".

## Usage

1. build dictiory
```python
python ./0_build_dict/build_dictionary.py --config ./configs/configs.yaml
```

2. binarise data
```python
python ./1_data_binarisation/binarise.py --config ./configs/configs.yaml
```

3. [Test Unit] Dataloader
```python
python ./2_test_dataloader/dataloader.py --config ./configs/configs.yaml
```

4. training
```python
python ./3_train_bart/train.py --config ./configs/configs.yaml
```

5. inference
```python
python ./4_infer_bart/inference.py --config ./configs/configs.yaml
```