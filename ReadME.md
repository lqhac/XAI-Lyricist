## Bart Usage

1. build dictiory  
python3 ./bartprompt/build_dictionary.py --config ./bartprompt/configs/bartprompt.yaml
python3 ./bartprompt/build_tokenizer.py --config ./bartprompt/configs/bartprompt.yaml

2. binarise data  
python3 ./bartprompt/corpus_compile_bart2.py --config ./bartprompt/configs/bartprompt-sentence.yaml
python3 ./bartprompt/corpus_compile_bart_whole.py --config ./bartprompt/configs/bartprompt-nokey.yaml
python3 ./bartprompt/corpus_compile_bart_whole_nokey.py --config ./bartprompt/configs/bartprompt-nokey.yaml
python3 ./bartprompt/corpus_compile_bart_whole_sep.py --config ./bartprompt/configs/bartprompt.yaml
python3 ./bartprompt/corpus_compile_bart_syllable.py --config ./bartprompt/configs/bartprompt-syllable.yaml
python3 ./bartprompt/corpus_compile_bart_free.py --config ./bartprompt/configs/bartprompt-free.yaml

## process custom data
python3 ./bartprompt/1_data_binarisation/corpus_compile_bart_whole_nokey.py --config ./bartprompt/configs/bartprompt-custom.yaml
python ./bartprompt/infer_bart_single_whole-nokey.py --config ./bartprompt/configs/bartprompt-custom.yaml
python ./bartprompt/infer_bart_single_whole-nokey-docx.py --config ./bartprompt/configs/bartprompt-custom.yaml

## generate counterfactuals for custom data
python ./bartprompt/gen_custom_counter_data.py --config ./bartprompt/configs/bartprompt-custom-counter.yaml
python3 ./bartprompt/counterfact_corpus_compile_bart_whole_nokey.py --config ./bartprompt/configs/bartprompt-custom-counter.yaml
python ./bartprompt/infer_bart_single_whole-counter-nokey-docx.py --config ./bartprompt/configs/bartprompt-custom-counter.yaml

python ./bartprompt/gen_custom_data.py --config ./bartprompt/configs/bartprompt-custom.yaml
python ./bartprompt/corpus_compile_bart_whole.py --config ./bartprompt/configs/configs/bartprompt-custom.yaml
python ./bartprompt/infer_bart_single_whole.py --config ./bartprompt/configs/bartprompt-custom.yaml

## process custom data in free mode
python ./bartprompt/corpus_compile_bart_free.py --config ./bartprompt/configs/bartprompt-free.yaml
python ./bartprompt/infer_bart_single_whole_free.py --config ./bartprompt/configs/bartprompt-free.yaml

  
3. [Test Unit] Dataloader
python3 ./bartprompt/dataloader.py --config ./bartprompt/configs/bartprompt.yaml

4. train tfmmodel 
python ./bartprompt/train_bart_con.py --config ./bartprompt/configs/bartprompt-sentence.yaml
python ./bartprompt/train_bart_con.py --config ./bartprompt/configs/bartprompt-sentence-large.yaml
python ./bartprompt/train_bart_con.py --config ./bartprompt/configs/bartprompt.yaml
python ./bartprompt/train_bart_con.py --config ./bartprompt/configs/bartprompt-nokey.yaml
python ./bartprompt/train_bart_con.py --config ./bartprompt/configs/bartprompt-large.yaml
python ./bartprompt/train_bart_con.py --config ./bartprompt/configs/bartprompt-syllable.yaml
python ./bartprompt/train_bart_con.py --config ./bartprompt/configs/bartprompt-free.yaml

5. inference tfmmodel  
python ./bartprompt/infer_bart.py --config ./bartprompt/configs/bartprompt.yaml
python ./bartprompt/infer_bart_single_sent.py --config ./bartprompt/configs/bartprompt.yaml
python ./bartprompt/infer_bart_single_whole.py --config ./bartprompt/configs/bartprompt-custom.yaml
python ./bartprompt/infer_bart_single_whole.py --config ./bartprompt/configs/bartprompt-large.yaml
python ./bartprompt/infer_bart_whole.py --config ./bartprompt/configs/bartprompt.yaml
python ./bartprompt/infer_bart_whole.py --config ./bartprompt/configs/bartprompt-syllable.yaml
python ./bartprompt/infer_bart_cond_prompt.py --config ./bartprompt/configs/bartprompt-sentence.yaml

6. saliency
python ./bartprompt/saliency_bart.py --config ./bartprompt/configs/bartprompt.yaml

7. calculate perplexity
python ./bartprompt/perplexity_bart.py --config ./bartprompt/configs/bartprompt.yaml

8. user study evaluation