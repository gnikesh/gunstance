
ROOT_DIR="/path/to/data"
RESULTS_DIR="/output/path"
AUGMENTATIONS_FILE="/path/to/augmentations.csv" # path to the csv containing the augmentations

UNLABELED_DATASET="unlabeled_restricted" # name of the csv containing the unlabeled data

while getopts s:h:t:e:f:i:d:b:r: flag
do
    case "${flag}" in
        s) SOFT_AUG=${OPTARG};;
        h) HARD_AUG=${OPTARG};;
        t) TASK=${OPTARG};;
        e) EVAL_TASK=${OPTARG};;
        f) FINAL_TASK=${OPTARG};;
        i) INDEX=${OPTARG};;
        d) DATE=${OPTARG};;
        b) ROOT_DIR=${OPTARG};;
        r) RESULTS_DIR=${OPTARG};;
    esac
done


echo "==================================================="
echo "$TASK    $EVAL_TASK    $FINAL_TASK        $SOFT_AUG    $HARD_AUG        $DATE    $INDEX"
echo "==================================================="

python textTrainHF.py \
    --threshold 0.7 \
    --epochs 100 \
    --batch-size 8 \
    --gradient_accumulation_steps 2 \
    --n_workers 1 \
    --patience 25 \
    --text_soft_aug "$SOFT_AUG" \
    --text_hard_aug "$HARD_AUG" \
    --mu 7 \
    --lambda-u 1 \
    --model "hf" \
    --hf_model "cardiffnlp/twitter-roberta-base-sep2022" \
    --data_path "$ROOT_DIR" \
    --task "$TASK" \
    --eval_task "$FINAL_TASK" \
    --train_file "train" \
    --unlabeled_dataset "$UNLABELED_DATASET" \
    --label_column "LabelStr" \
    --text_column "Text" \
    --query_column "Query" \
    --augmentations_file "$AUGMENTATIONS_FILE" \
    --resume "${RESULTS_DIR}/${TASK}_${EVAL_TASK}_${SOFT_AUG}_${HARD_AUG}_lu1_mu7_ema0_${DATE}_${INDEX}/model_best.pth.tar" \
    --eval_only
