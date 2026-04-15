import os
import time
import torch
import argparse
import numpy as np
from loguru import logger
from datetime import datetime
from transformers import set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from src.dataset import PIDataset, JCDataset, SGDataset, collate_fn
from src.model import PteModel, PteCriterion
from src.utils import stats_time
from src.eval import evaluation


DATASET_CLASSES = {
    'PI': PIDataset,
    'JC': JCDataset,
    'SG': SGDataset,
}



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=333, help="random seed for initialization.")
    parser.add_argument('--batch_size', default=8, type=int, help="Total batch size for training.")
    parser.add_argument('--epochs', default=2000, type=int, help='The epoch of train')
    parser.add_argument('--model_path', required=True, type=str, help='The pretrained model')
    parser.add_argument('--task_name', required=True, type=str, help='The name of task')
    parser.add_argument('--max_seq_length', default=512, type=int, help="The maximum length of squence")
    parser.add_argument('--train_data_path', type=str, required=True, help="The path of train Toxic Comment Classification Challenge dataset")
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpts', help="The directory of checkpoints")
    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard', help="The directory of tensorboard")
    parser.add_argument('--log_dir', type=str, default='./log', help="The directory of log")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help="The step of gradient accumulation")
    parser.add_argument('--learning_rate', default=5e-5, type=float, help="The initial learning rate for optimizer")
    parser.add_argument('--eval_data_path', type=str, default=None, help="The path of eval dataset (optional, enables validation if provided)")
    parser.add_argument('--test_data_path', type=str, default=None, help="The path of test dataset (optional, used before final checkpoint save)")
    parser.add_argument('--eval_freq', default=500, type=int, help='The freq of eval test set')
    parser.add_argument('--log_freq', default=100, type=int, help='The freq of print log')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="The adam epsilon")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help="The maximum gradient normalization")
    parser.add_argument('--warmup_steps', default=0, type=float, help="The steps of  warm up")
    parser.add_argument('--lr_scheduler', default='cosine', type=str, choices=['linear', 'cosine', 'constant', 'polynomial'],
                        help="Learning rate scheduler type: linear, cosine, constant, polynomial")
    parser.add_argument('--weight_lr', default=0.55, type=float, help="The learning rate of words' weight")
    parser.add_argument('--classes_num', default=2, type=int, help="The number of labels")
    parser.add_argument('--pattern_ids', default=3, type=int, help="The ids of pattern")
    args = parser.parse_args()

    if args.eval_data_path is None and args.test_data_path is None:
        raise ValueError("Either --eval_data_path or --test_data_path must be provided.")

    return args


def setup_training(config):
    # config.seed = np.random.randint(1, 100000)
    set_seed(config.seed)
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    config.device = device

    # tensorboard set up
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    comment = f'bath_size={config.batch_size} lr={config.learning_rate}'
    writer = SummaryWriter(log_dir=config.tensorboard_dir + "/" + time_stamp, comment=comment)

    # cuda setup
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # hidden tokenizer warning
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    # logger setup
    logger.add(os.path.join(config.log_dir) + "/" +"{time}.log")

    # checkpoint dir setup
    checkpoint_dir = os.path.join(config.checkpoint_dir, time_stamp)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    config.checkpoint_dir = checkpoint_dir

    return config, writer

def prepare_data_loader(config, num_workers=1):
    dataset_cls = DATASET_CLASSES.get(config.task_name)
    if dataset_cls is None:
        raise ValueError(f"Unsupported task name: {config.task_name}")

    train_dataset = dataset_cls(config.train_data_path, config.model_path, config.pattern_ids, config.max_seq_length)
    train_data_iter = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config.batch_size, num_workers=num_workers, shuffle=True)

    eval_data_iter = None
    if config.eval_data_path is not None:
        eval_dataset = dataset_cls(config.eval_data_path, config.model_path, config.pattern_ids, config.max_seq_length)
        eval_data_iter = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=config.batch_size, num_workers=num_workers, shuffle=False)

    test_data_iter = None
    if config.test_data_path is not None:
        test_dataset = dataset_cls(config.test_data_path, config.model_path, config.pattern_ids, config.max_seq_length)
        test_data_iter = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=config.batch_size, num_workers=num_workers, shuffle=False)

    return train_data_iter, eval_data_iter, test_data_iter, train_dataset.m2c_tensor, train_dataset.filler_len

def prepare_model_and_optimizer(config, m2c_tensor, filler_len, total_step):
    model = PteModel(config)
    model.to(config.device)
    criterion = PteCriterion(config, m2c_tensor, filler_len)
    criterion.to(config.device)
    model_parameters = []
    other_parameters = []
    for name, param in model.named_parameters():
        if name.startswith('model'):
            model_parameters.append(param)
        else:
            other_parameters.append(param)

    optimizer = AdamW([{'params':model_parameters, 'lr': config.learning_rate},
                       {'params':other_parameters, 'lr': config.weight_lr}], lr=0, eps=config.adam_epsilon, no_deprecation_warning=True)

    if config.lr_scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_step)
    elif config.lr_scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_step)
    elif config.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps)
    elif config.lr_scheduler == 'polynomial':
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_step)
    else:
        raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")

    return model, criterion, optimizer, scheduler

def trainer():
    config = parse_arguments()
    config, writer = setup_training(config)
    train_iter, eval_iter, test_iter, m2c_tensor, filler_len = prepare_data_loader(config)
    total_step = config.epochs * len(train_iter)

    model, criterion, optimizer, scheduler = prepare_model_and_optimizer(config, m2c_tensor, filler_len, total_step)

    logger.info(f"{'#' * 41} Config {'#' * 41}")
    for k in list(vars(config).keys()):
        logger.info('{0}: {1}'.format(k, vars(config)[k]))
    logger.info(f'total step: {total_step}')
    logger.info(f'the number of train step: {len(train_iter)}')
    logger.info(f'the size of train set: {len(train_iter.dataset)}')
    if eval_iter is not None:
        logger.info(f'the size of eval set: {len(eval_iter.dataset)}')
    if test_iter is not None:
        logger.info(f'the size of test set: {len(test_iter.dataset)}')
    logger.info(f"{'#' * 41} Training {'#' * 41}")

    start = int(time.time())
    step = 0
    avg_loss = 0.0
    has_eval = eval_iter is not None
    for epoch in range(1, config.epochs+1):
        # train
        for i, batch in enumerate(train_iter):
            model.train()
            step += 1
            input_ids, token_type_ids, attention_mask, mlm_labels, labels, _ = [
                w.to(config.device) if isinstance(w, torch.Tensor) else w for w in batch
            ]
            logit, weight = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss, predictions = criterion(logit, mlm_labels, labels, weight)

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            avg_loss += loss.item()
            if step % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


            # tensorboard
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('avg_loss', avg_loss / step, step)

            # log
            if step % config.log_freq == 0:
                end = int(time.time())
                logger.info(f"epochs:{str(epoch) + '/' + str(config.epochs)}, batch:{str(i + 1) + '/' + str(len(train_iter))}, step:{str(step) + '/' + str(total_step)}, cur_loss:{'{:.6f}'.format(loss)}, avg_loss:{'{:.6f}'.format(avg_loss/step)}, eta:{stats_time(start, end, step, total_step)}h")

            # evaluation
            if has_eval and step % config.eval_freq == 0:
                metrics = evaluation(eval_iter, model, criterion, config.device)
                writer.add_scalar('acc', metrics['accuracy'], step)
                writer.add_scalar('macro_precision', metrics['macro_precision'], step)
                writer.add_scalar('macro_recall', metrics['macro_recall'], step)
                writer.add_scalar('macro_f1', metrics['macro_f1'], step)
                logger.info(
                    f"epochs:{str(epoch) + '/' + str(config.epochs)}, step:{str(step) + '/' + str(total_step)}, "
                    f"acc:{metrics['accuracy']:.6f}, macro_precision:{metrics['macro_precision']:.6f}, "
                    f"macro_recall:{metrics['macro_recall']:.6f}, macro_f1:{metrics['macro_f1']:.6f}"
                )

    final_eval_iter = test_iter if test_iter is not None else eval_iter
    final_eval_name = 'test' if test_iter is not None else 'eval'
    if final_eval_iter is not None:
        final_metrics = evaluation(final_eval_iter, model, criterion, config.device)
        writer.add_scalar(f'final_{final_eval_name}_acc', final_metrics['accuracy'], step)
        writer.add_scalar(f'final_{final_eval_name}_macro_precision', final_metrics['macro_precision'], step)
        writer.add_scalar(f'final_{final_eval_name}_macro_recall', final_metrics['macro_recall'], step)
        writer.add_scalar(f'final_{final_eval_name}_macro_f1', final_metrics['macro_f1'], step)
        logger.info(
            f"final {final_eval_name} before saving checkpoint, acc:{final_metrics['accuracy']:.6f}, "
            f"macro_precision:{final_metrics['macro_precision']:.6f}, "
            f"macro_recall:{final_metrics['macro_recall']:.6f}, macro_f1:{final_metrics['macro_f1']:.6f}"
        )

    # save checkpoint after training finishes
    final_model_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    final_weight_path = os.path.join(config.checkpoint_dir, "final_model_weight.pt")
    torch.save(model.model.state_dict(), final_model_path)
    torch.save({'weight': model.weight}, final_weight_path)
    logger.info(f"Training finished. Model saved to {final_model_path}")


if __name__ == '__main__':
    trainer()
