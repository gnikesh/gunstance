RESULTS_DIR="/output/path/experiment2"

declare -a SOFT_AUGS=("TW~0~2")
declare -a HARD_AUGS=("BT~3~25")
declare -a INDEXES=("0" "1" "2")

declare -a TASKS=("ban_all7-buffalo" "ban_all7-uvalde" "ban_all7-boulder" "regulate_all7-buffalo" "regulate_all7-uvalde" "regulate_all7-boulder")
declare -a EVAL_TASKS=("ban_all7" "ban_all7" "ban_all7" "regulate_all7" "regulate_all7" "regulate_all7")

DATE="12oct"


for INDEX in "${INDEXES[@]}"
do
    for SOFT_AUG in "${SOFT_AUGS[@]}"
    do
        for HARD_AUG in "${HARD_AUGS[@]}"
        do
            for i in "${!TASKS[@]}"
            do
                TASK="${TASKS[$i]}"
                EVAL_TASK="${EVAL_TASKS[$i]}"

                echo "A pause of 5m will be made after each experiment"
                ./training_scripts/fixmatch_train_experiment.sh -s $SOFT_AUG -h $HARD_AUG -t $TASK -e $EVAL_TASK -i $INDEX -d $DATE -r $RESULTS_DIR
                echo "Waiting 5m before the next experiment"
                sleep 5m
                echo "Done"
            done
        done
    done
done
