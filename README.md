# XAI-Lyricist
The official repository for the paper ***[XAI-Lyricist: Improving the Singability of AI-Generated Lyrics with Prosody Explanations](https://www.ijcai.org/proceedings/2024/0872)*** by Qihao Liang, Xichu Ma, Finale Doshi-Velez, Brian Lim, and Ye Wang. This paper has been published at the *[the 33th International Joint Conference on Artificial Intelligence (IJCAI 2024), Special Track on Human-Centred Artificial Intelligence: Multidisciplinary Contours and Challenges of Next-Generation AI Research and Applications](https://ijcai24.org/call-for-papers-human-centred-artificial-intelligence/), 3rd-9th August, Jeju, South Korea.** 

## Usage

### STEP 0: Environmental Setup
```shell
export PYTHONPATH=.
```

### STEP 1: Building Dictionaries
We first construct dictionaries for both lyrics and melodies with the following code. 
```python
python ./0_build_dict/build_dictionary.py -config ./configs/configs.yaml
```

### STEP 2: Binarising Data
With the dictionaries ready, we binarise the dataset by converting lyrics and melodies to tokens.
```python
python ./1_data_binarisation/binarise.py -config ./configs/configs.yaml
```

### STEP 3: Training
```python
python ./3_train_bart/train.py -config ./configs/configs.yaml
```
### STEP 4: Inference
We provide two versions of lyrics inference, **melody-based** and **parody-based**.
#### Melody-Based Lyrics Generation
For melody-based inference, the input is a MIDI file with melody phrase boundaries marked. `imagine_midi_test.mid` provides a good example. The MIDI is first analysed and converted to a prosody template conditioning lyrics generation. The resulting lyrics are expected to share the same prosodic pattern as the melody.
```python
python ./4_infer_bart/inference.py -config ./configs/configs.yaml
```
#### Parody-Based Lyrics Generation
The parody-based inference uses lyrics as input. The system analyses the prosody of lyrics by retrieving their IPA annotation, marking each syllable with a strength and length symbol. This prosody further conditions the model to generate a new piece of lyrics with the same prosodic pattern as the input.
```
python ./4_infer_bart/inference_parody.py -config ./configs/configs.yaml
```


### To cite this work
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