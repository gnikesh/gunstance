# GunStance

This is the code and data for our ACL 2024 paper entitled 
[GunStance: Stance Detection for Gun Control and Gun Regulation](https://aclanthology.org/2024.acl-long.650/).

If you use the GunStance dataset or the code from this repository in your research, please cite our paper:
```bibtex
@inproceedings{gyawali-etal-2024-gunstance,
  title={GunStance: Stance Detection for Gun Control and Gun Regulation},
  author={Gyawali, Nikesh and Sirbu, Iustin and Sosea, Tiberiu and Khanal, Sarthak and Caragea, Doina and Rebedea, Traian and Caragea, Cornelia},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month = aug,
  year = "2024",
  address = "Bangkok, Thailand",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2024.acl-long.650",
  pages = "12027--12044",
}
```

## Abstract
The debate surrounding gun control and gun regulation in the United States has intensified in the wake of numerous mass shooting events. As perspectives on this matter vary, it becomes increasingly important to comprehend individuals’ positions. Stance detection, the task of determining an author’s position towards a proposition or target, has gained attention for its potential use in understanding public perceptions towards controversial topics and identifying the best strategies to address public concerns. In this paper, we present GunStance, a dataset of tweets pertaining to shooting events, focusing specifically on the controversial topics of “banning guns” versus “regulating guns.” The tweets in the dataset are sourced from discussions on Twitter following various shooting incidents in the United States. Amazon Mechanical Turk was used to manually annotate a subset of the tweets relevant to the targets of interest (“banning guns” and “regulating guns”) into three classes: In-Favor, Against, and Neutral. The remaining unlabeled tweets are included in the dataset to facilitate studies on semi-supervised learning (SSL) approaches that can help address the scarcity of the labeled data in stance detection tasks. Furthermore, we propose a hybrid approach that combines curriculum-based SSL and Large Language Models (LLM), and show that the proposed approach outperforms supervised, semi-supervised, and LLM-based zero-shot models in most experiments on our assembled dataset.

## Dataset

The GunStance dataset, containing all the tweet ids (including duplicate ids for the tweets used for both queries), can be found at `dataset/GunStanceDataset.csv`.

In the folder `dataset/splits` we also provide the dataset in the structure used by this code. For example, in the case of our leave-one-out experiments, the code expects the train data from the `ban_all7-buffalo` folder and the test data from the `ban_buffalo` folder. This structure can be directly obtained from the full dataset by filtering the tweets corresponding to the desired events (using the `Event` column) and by splitting the tweets in the corresponding files (labeled/unlabeled/valid/test) by the values of the `Split` column.

## Code References

[AUM-ST](https://github.com/tsosea2/AUM-ST)

[FixMatchLS](https://github.com/iustinsirbu13/multimodal-ssl-for-disaster-tweet-classification)
