import os
import json
import textwrap


def save_flags(FLAGS):
    save_dir = os.path.join(FLAGS.base_dir, FLAGS.hyperparameters_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    flags_dict = {
        "pt_teacher_checkpoint": FLAGS.pt_teacher_checkpoint,
        "saved_teacher_model_path": FLAGS.saved_teacher_model_path,
        "base_dir": FLAGS.base_dir,
        "augmentation_dir": FLAGS.augmentation_dir,
        "dataset_path": FLAGS.dataset_path,
        "validation_path": FLAGS.validation_path,
        "test_path": FLAGS.test_path,
        "train_path": FLAGS.train_path,
        "unlabelled_path": FLAGS.unlabelled_path,
        "num_labels": FLAGS.num_labels,
        "max_seq_len": FLAGS.max_seq_len,
        "num_classes": FLAGS.num_classes,
        "seed": FLAGS.seed,
        "weak_augmentation_min_strength": FLAGS.weak_augmentation_min_strength,
        "weak_augmentation_max_strength": FLAGS.weak_augmentation_max_strength,
        "strong_augmentation_min_strength": FLAGS.strong_augmentation_min_strength,
        "strong_augmentation_max_strength": FLAGS.strong_augmentation_max_strength,
        "sup_batch_size": FLAGS.sup_batch_size,
        "unsup_batch_size": FLAGS.unsup_batch_size,
        "inference_batch_size": FLAGS.inference_batch_size,
        "initial_num": FLAGS.initial_num,
        "intermediate_model_path": FLAGS.intermediate_model_path,
        "supervised_patience": FLAGS.supervised_patience,
        "unsupervised_patience": FLAGS.unsupervised_patience,
        "self_training_steps": FLAGS.self_training_steps,
        "unlabeled_epochs_per_step": FLAGS.unlabeled_epochs_per_step,
        "supervised_once_epochs": FLAGS.supervised_once_epochs,
        "threshold": FLAGS.threshold,
        "aum_percentile": FLAGS.aum_percentile,
        "tensorboard_dir": FLAGS.tensorboard_dir,
        "aum_save_dir": FLAGS.aum_save_dir,
        "hyperparameters_save_dir": FLAGS.hyperparameters_save_dir,
        "experiment_id": FLAGS.experiment_id,
        "gpu_id": FLAGS.gpu_id
    }

    pprint_flag_dict = json.dumps(flags_dict, indent=2).splitlines()
    pprint_flag_dict = '\n'.join([pprint_flag_dict[0], textwrap.dedent('\n'.join(pprint_flag_dict[1:-1])), pprint_flag_dict[-1]])

    with open(os.path.join(save_dir, 'hyperparameters.json'), 'w') as fwrite:
        fwrite.write(pprint_flag_dict)

    print(f"written parameters to {save_dir}")


def save_results_dict(results_dict):
    pass
