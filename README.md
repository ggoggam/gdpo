## GDPO: GFlowNet Direct Preference Optimization

This is a repository containing the code for [GDPO: Learning to Directly Align Language Models with Diversity Using GFlowNets](https://aclanthology.org/2024.emnlp-main.951.pdf), published in EMNLP 2024 Main.

### 🚧 Currently Under Construction
This code base is being organized for ease of use.

### Training
Make sure you modify the appropriate `accelerate` config located in `config/accelerate` directory according to your machine configuration. From the `/src` directory, run training by one of the following commands with a choice of machine type.

```shell
accelerate launch --config-file config/accelerate/{type}.yaml train.py
```
```shell
python -u -m accelerate.commands.launch --config-file conifg/accelerate/{type}.yaml train.py
```

For now, we only provide offline training, which was the focus of the paper.

### Evaluating

### Reference

```
@inproceedings{kwon-etal-2024-gdpo,
    title = "{GDPO}: Learning to Directly Align Language Models with Diversity Using {GF}low{N}ets",
    author = "Kwon, Oh Joon  and
      Matsunaga, Daiki E.  and
      Kim, Kee-Eung",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.951",
    pages = "17120--17139",
}
```
