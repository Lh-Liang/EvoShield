import os
import time
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from loguru import logger
from datetime import datetime
from collections import deque
from transformers import set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.dataset import PIDataset, JCDataset, SGDataset, collate_fn
from src.model import PteModel, PteCriterion
from src.utils import stats_time
from src.llm_client import LLMClient



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization.")
    parser.add_argument('--batch_size', default=32, type=int, help="Total batch size for training.")
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to train')
    parser.add_argument('--model_path', default='/home/han/llh/EvoShield/bert-base-uncased', required=False, type=str, help='The pretrained model')
    parser.add_argument('--task_name', default='JC', required=False, type=str, help='The name of task')
    parser.add_argument('--max_seq_length', default=512, type=int, help="The maximum length of squence")
    parser.add_argument('--train_data_path', type=str, default='/home/zxh/code/EvoShield/data/jailbreak-classification-balanced/full_test.csv', required=False, help="The path of train Toxic Comment Classification Challenge dataset")
    parser.add_argument('--checkpoint_dir', type=str, default='/home/zxh/code/EvoShield/ckpts', help="The directory of checkpoints")
    parser.add_argument('--tensorboard_dir', type=str, default='/home/zxh/code/EvoShield/tensorboard', help="The directory of tensorboard")
    parser.add_argument('--log_dir', type=str, default='/home/zxh/code/EvoShield/log', help="The directory of log")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="The step of gradient accumulation")
    parser.add_argument('--learning_rate', default=5e-5, type=float, help="The initial learning rate for optimizer")
    parser.add_argument('--log_freq', default=10, type=int, help='The freq of print log')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="The adam epsilon")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help="The maximum gradient normalization")
    parser.add_argument('--warmup_steps', default=0, type=float, help="The steps of  warm up")
    parser.add_argument('--lr_scheduler', default='cosine', type=str, choices=['linear', 'cosine', 'constant', 'polynomial'],
                        help="Learning rate scheduler type: linear, cosine, constant, polynomial")
    parser.add_argument('--weight_lr', default=0.55, type=float, help="The learning rate of words' weight")
    parser.add_argument('--classes_num', default=2, type=int, help="The number of labels")
    parser.add_argument('--pattern_ids', default=3, type=int, help="The ids of pattern")
    # LLM related parameters
    parser.add_argument('--use_llm', default=True, type=bool, help="Whether to use LLM for entropy threshold filtering")
    parser.add_argument('--entropy_threshold', default=0.2, type=float, help="Entropy threshold, above which LLM is called")
    parser.add_argument('--model', default='gemini-3.1-pro-preview', type=str, help="The name of LLM")
    # Review training related parameters
    parser.add_argument('--review_window_size', default=128, type=int, help="Sliding window size, keep recent n high-entropy samples")
    parser.add_argument('--review_freq', default=1, type=int, help="Review frequency (every n steps)")
    # Validation related parameters
    parser.add_argument('--val_data_path', type=str, default=None, help="The path of validation dataset")
    parser.add_argument('--val_freq', default=3, type=int, help="Validation frequency (every n epochs)")
    parser.add_argument('--metric_for_best', type=str, default='accuracy', choices=['accuracy', 'precision', 'recall', 'f1'], help="Metric for selecting best model")
    args = parser.parse_args()

    return args


def compute_entropy(logits: torch.Tensor, mlm_labels: torch.Tensor, m2c, filler_len, weight) -> torch.Tensor:
    """
    Compute entropy of class probability distribution

    Args:
        logits: MLM output logits [batch, seq_len, vocab_size]
        mlm_labels: mask position indicators
        m2c: class to verbalizer mapping tensor
        filler_len: number of verbalizers per class
        weight: class weights

    Returns:
        entropy tensor [batch]
    """
    # Extract logits at masked positions
    masked_logits = logits[mlm_labels >= 0]  # [batch, vocab_size]

    # Convert MLM logits to class logits
    cls_logits_list = []
    for ml in masked_logits:
        # Use the same conversion logic as criterion
        m2c_filtered = torch.max(torch.zeros_like(m2c), m2c)
        cls_logits = ml[m2c_filtered]
        cls_logits = cls_logits * (m2c > 0).float()
        cls_logits = (weight * cls_logits).sum(axis=1) / filler_len
        cls_logits_list.append(cls_logits)

    cls_logits = torch.stack(cls_logits_list)  # [batch, num_classes]

    # Compute class probability distribution
    probs = F.softmax(cls_logits, dim=-1)

    # Compute entropy
    eps = 1e-10
    probs = torch.clamp(probs, min=eps, max=1-eps)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)

    return entropy


def init_llm_client(config):
    """
    Initialize LLM client

    Args:
        config: configuration object

    Returns:
        LLMClient instance or None
    """
    if hasattr(config, 'use_llm') and config.use_llm:
        return LLMClient(config.model)
    return None


def get_task_categories(task_name: str) -> dict:
    """
    Get task category mapping

    Args:
        task_name: task name

    Returns:
        category mapping dictionary
    """
    categories_map = {
        "PI": {
            0: "benign",
            1: "injection"
        },
        "JC": {
            0: "benign",
            1: "jailbreak"
        },
        "SG": {
            0: "benign",
            1: "injection"
        }
    }
    return categories_map.get(task_name, None)


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
    train_dataset = eval(config.task_name + 'Dataset')(config.train_data_path, config.model_path, config.pattern_ids, config.max_seq_length)
    train_data_iter = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config.batch_size, num_workers=num_workers, shuffle=True)

    return train_data_iter, train_dataset.m2c_tensor, train_dataset.filler_len


def prepare_val_data_loader(config, num_workers=1):
    """
    Prepare validation data loader

    Args:
        config: configuration object
        num_workers: number of workers for data loading

    Returns:
        val_data_iter: validation data iterator
    """
    if config.val_data_path is None:
        return None, None, None

    val_dataset = eval(config.task_name + 'Dataset')(config.val_data_path, config.model_path, config.pattern_ids, config.max_seq_length)
    val_data_iter = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=config.batch_size, num_workers=num_workers, shuffle=False)

    return val_data_iter, val_dataset.m2c_tensor, val_dataset.filler_len

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
    
    # Select learning rate decay strategy
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

def compute_classification_metrics(all_predictions, all_labels):
    """
    Compute classification metrics

    Args:
        all_predictions: list of predicted labels
        all_labels: list of true labels

    Returns:
        dictionary containing various metrics
    """
    acc = accuracy_score(all_labels, all_predictions)
    macro_precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    return {
        'accuracy': acc,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1
    }


def validate(model, criterion, val_iter, config):
    """
    Evaluate model on validation set (using trained model for prediction directly, without computing entropy or calling LLM)

    Args:
        model: model
        criterion: loss function
        val_iter: validation data iterator
        config: configuration object

    Returns:
        metrics: validation metrics dictionary
    """
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_iter:
            input_ids, token_type_ids, attention_mask, mlm_labels, original_labels, texts = [
                w.to(config.device) if isinstance(w, torch.Tensor) else w for w in batch
            ]

            logit, weight = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            # Use model predictions directly
            masked_logits = logit[mlm_labels >= 0]
            cls_logits = torch.stack([
                criterion._convert_single_mlm_logits_to_cls_logits(ml, weight)
                for ml in masked_logits
            ])
            predictions = criterion.predict(cls_logits)

            all_predictions.extend(predictions)
            all_labels.extend(original_labels.cpu().numpy())

    # Compute metrics
    metrics = compute_classification_metrics(all_predictions, all_labels)

    return metrics


def prepare_review_batch(review_texts, review_labels, dataset, device):
    """
    Convert window samples to model input

    Args:
        review_texts: list of texts
        review_labels: list of labels
        dataset: training dataset (used to get tokenizer and encode method)
        device: torch.device

    Returns:
        batch: (input_ids, token_type_ids, attention_mask, mlm_labels, labels_tensor, texts)
    """
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    mlm_labels_list = []

    for text in review_texts:
        # Use dataset's encode method for tokenization
        feature = dataset.encode(text)
        input_ids = feature.input_ids.squeeze(0)
        token_type_ids = feature.token_type_ids.squeeze(0)
        attention_mask = feature.attention_mask.squeeze(0)

        # Get mlm_labels
        mlm_labels = dataset.get_mlm_labels(input_ids.tolist())

        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
        mlm_labels_list.append(mlm_labels)

    # Manual collate
    input_ids = torch.stack(input_ids_list)
    token_type_ids = torch.stack(token_type_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    mlm_labels = torch.stack([torch.tensor(ml, dtype=torch.long) for ml in mlm_labels_list])
    labels_tensor = torch.tensor(review_labels, dtype=torch.long).unsqueeze(1)

    # Move to device
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    mlm_labels = mlm_labels.to(device)
    labels_tensor = labels_tensor.to(device)

    return input_ids, token_type_ids, attention_mask, mlm_labels, labels_tensor, review_texts


def review_training(model, criterion, review_window, dataset, config, optimizer):
    """
    Perform batch-wise review training with window samples

    Args:
        model: PteModel
        criterion: PteCriterion
        review_window: deque storing (text, llm_label)
        dataset: training dataset (used to get tokenizer)
        config: configuration object
        optimizer: optimizer (used for parameter update)

    Returns:
        avg_loss: average loss
    """
    if len(review_window) == 0:
        return 0.0

    model.train()

    texts = [item[0] for item in review_window]
    labels = [item[1] for item in review_window]

    total_loss = 0.0
    num_batches = 0

    # Process in batches (using batch_size)
    for i in range(0, len(texts), config.batch_size):
        batch_texts = texts[i:i + config.batch_size]
        batch_labels = labels[i:i + config.batch_size]

        # Prepare data
        input_ids, token_type_ids, attention_mask, mlm_labels, batch_labels_tensor, _ = \
            prepare_review_batch(batch_texts, batch_labels, dataset, config.device)

        # Forward pass
        logit, weight = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # Compute loss (using LLM labels)
        loss, _ = criterion(logit, mlm_labels, batch_labels_tensor, weight)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        # Update parameters separately
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def trainer():
    config = parse_arguments()
    config, writer = setup_training(config)
    train_iter, m2c_tensor, filler_len = prepare_data_loader(config)

    # Initialize LLM client
    llm_client = init_llm_client(config)
    if llm_client:
        logger.info(f"LLM client initialized, model: {config.model}, entropy_threshold: {config.entropy_threshold}")
        task_categories = get_task_categories(config.task_name)
        if task_categories is None:
            logger.warning(f"Category mapping not found for task {config.task_name}, using default categories")
            task_categories = {i: str(i) for i in range(config.classes_num)}
    else:
        task_categories = None
        logger.info("LLM client not initialized, use_llm=False")

    # Initialize review training sliding window
    review_window = deque(maxlen=config.review_window_size)
    logger.info(f"Review window initialized, window_size={config.review_window_size}, review_freq={config.review_freq}")

    # Prepare validation data loader
    val_iter = None
    if config.val_data_path is not None:
        val_iter, val_m2c_tensor, val_filler_len = prepare_val_data_loader(config)
        logger.info(f"Validation enabled: {config.val_data_path}")
        logger.info(f"Validation samples: {len(val_iter.dataset)}")
    else:
        logger.info("Validation disabled (val_data_path not specified)")

    # Calculate total steps: epochs × len(train_iter)
    total_step = config.epochs * len(train_iter)

    # Create model and optimizer
    model, criterion, optimizer, scheduler = prepare_model_and_optimizer(config, m2c_tensor, filler_len, total_step)

    logger.info(f"{'#' * 41} Config {'#' * 41}")
    for k in list(vars(config).keys()):
        logger.info('{0}: {1}'.format(k, vars(config)[k]))
    logger.info(f'total step: {total_step}')
    logger.info(f'the number of train step: {len(train_iter)}')
    logger.info(f'the size of train set: {len(train_iter.dataset)}')
    logger.info(f"{'#' * 41} Training {'#' * 41}")

    start = int(time.time())
    step = 0
    avg_loss = 0.0

    # Validation related variables
    best_metric_value = 0.0
    best_epoch = 0

    # Accumulate LLM and Model predictions and labels (for computing overall metrics after training)
    llm_preds_all = []
    llm_labels_all = []
    model_preds_all = []
    model_labels_all = []

    for epoch in range(1, config.epochs + 1):

        for batch_idx, batch in enumerate(train_iter):
            model.train()
            input_ids, token_type_ids, attention_mask, mlm_labels, original_labels, texts = [w.to(config.device) if isinstance(w, torch.Tensor) else w for w in batch]

            step += 1

            # Forward pass
            logit, weight = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            # Compute class probability distribution and entropy
            with torch.no_grad():
                entropies = compute_entropy(logit, mlm_labels, criterion.m2c, criterion.filler_len, weight)

            # Initialize labels (copy from original_labels)
            labels = original_labels.clone()

            # Handle high-entropy samples - LLM prediction
            if llm_client is not None:
                # Find high-entropy samples in the batch
                llm_mask = entropies > config.entropy_threshold
                high_entropy_indices = torch.where(llm_mask)[0].cpu().tolist()
                low_entropy_indices = [i for i in range(len(entropies)) if i not in high_entropy_indices]

                # Set low-entropy sample labels to -100 to be ignored by loss function
                if len(low_entropy_indices) > 0:
                    labels[low_entropy_indices] = -100

                # Call LLM to predict high-entropy samples
                if len(high_entropy_indices) > 0:
                    for idx in high_entropy_indices:
                        text = texts[idx]
                        try:
                            llm_pred = llm_client.predict(text, config.task_name, task_categories)
                            if llm_pred is None:
                                labels[idx] = -100
                                logger.warning(f"Step {step}: LLM result unparseable, skip sample")
                                continue
                            labels[idx] = llm_pred
                            # Add high-entropy sample to review window
                            review_window.append((text, llm_pred))
                        except Exception as e:
                            logger.warning(f"Step {step}: LLM prediction failed, using original labels. Error: {e}")

                # Log entropy statistics
                avg_entropy = entropies.mean().item()
                max_entropy = entropies.max().item()
                high_entropy_count = len(high_entropy_indices)
                high_entropy_ratio = high_entropy_count / len(entropies) if len(entropies) > 0 else 0
                logger.info(f"[Epoch {epoch} Batch {batch_idx+1}] avg_entropy={avg_entropy:.4f}, max_entropy={max_entropy:.4f}, high_entropy_count={high_entropy_count}/{len(entropies)}, ratio={high_entropy_ratio:.2%}")

                # Log entropy statistics to tensorboard
                writer.add_scalar('entropy', avg_entropy, step)
                writer.add_scalar('high_entropy_ratio', high_entropy_ratio, step)

                # Compute classification metrics
                with torch.no_grad():
                    masked_logits = logit[mlm_labels >= 0]
                    cls_logits = torch.stack([
                        criterion._convert_single_mlm_logits_to_cls_logits(ml, weight)
                        for ml in masked_logits
                    ])
                    model_preds = criterion.predict(cls_logits)

                true_labels = original_labels.cpu().numpy()

                # High-entropy sample metrics (using LLM predictions)
                if len(high_entropy_indices) > 0:
                    valid_high_entropy_indices = [idx for idx in high_entropy_indices if labels[idx].item() != -100]
                    if len(valid_high_entropy_indices) > 0:
                        llm_preds = [labels[idx].item() for idx in valid_high_entropy_indices]
                        llm_true = [true_labels[idx] for idx in valid_high_entropy_indices]
                        llm_metrics = compute_classification_metrics(llm_preds, llm_true)

                        # Accumulate LLM predictions and labels
                        llm_preds_all.extend(llm_preds)
                        llm_labels_all.extend(llm_true)

                        writer.add_scalar('llm_accuracy', llm_metrics['accuracy'], step)
                        writer.add_scalar('llm_precision', llm_metrics['precision'], step)
                        writer.add_scalar('llm_recall', llm_metrics['recall'], step)
                        writer.add_scalar('llm_f1', llm_metrics['f1'], step)
                        logger.info(f"[High-Entropy] n={len(valid_high_entropy_indices)} | LLM preds: acc={llm_metrics['accuracy']:.4f}, precision={llm_metrics['precision']:.4f}, recall={llm_metrics['recall']:.4f}, f1={llm_metrics['f1']:.4f}")

                # Low-entropy sample metrics (using model predictions)
                if len(low_entropy_indices) > 0:
                    model_preds_list = [model_preds[idx] for idx in range(len(model_preds)) if idx in low_entropy_indices]
                    model_true = [true_labels[idx] for idx in range(len(true_labels)) if idx in low_entropy_indices]
                    model_metrics = compute_classification_metrics(model_preds_list, model_true)

                    # Accumulate Model predictions and labels
                    model_preds_all.extend(model_preds_list)
                    model_labels_all.extend(model_true)

                    writer.add_scalar('model_accuracy', model_metrics['accuracy'], step)
                    writer.add_scalar('model_precision', model_metrics['precision'], step)
                    writer.add_scalar('model_recall', model_metrics['recall'], step)
                    writer.add_scalar('model_f1', model_metrics['f1'], step)
                    logger.info(f"[Low-Entropy] n={len(low_entropy_indices)} | Model preds: acc={model_metrics['accuracy']:.4f}, precision={model_metrics['precision']:.4f}, recall={model_metrics['recall']:.4f}, f1={model_metrics['f1']:.4f}")

            # Compute loss
            if llm_client is None:
                # When not using LLM, use all samples
                loss, predictions = criterion(logit, mlm_labels, labels, weight)

                loss = loss / config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                avg_loss += loss.item() * config.gradient_accumulation_steps
                if step % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            elif len(high_entropy_indices) > 0:
                # When using LLM, only compute loss on high-entropy samples
                loss, predictions = criterion(logit, mlm_labels, labels, weight)

                # Keep only high-entropy sample predictions for evaluation
                predictions = predictions[high_entropy_indices]

                loss = loss / config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                avg_loss += loss.item() * config.gradient_accumulation_steps
                if step % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                # use_llm=true but no high-entropy samples, skip loss computation and parameter update
                loss = torch.tensor(0.0, device=config.device)
                # This batch does not contribute gradients, so release its forward graph
                # before review training to avoid stacking two MLM graphs in memory.
                del logit, weight

            # === Review Training ===
            # Force parameter update before review to avoid gradient confusion
            if llm_client is not None and len(review_window) > 0 and step % config.review_freq == 0:
                # If gradient accumulation is not complete, force parameter update first
                if step % config.gradient_accumulation_steps != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                review_loss = review_training(
                    model, criterion, review_window,
                    train_iter.dataset, config, optimizer
                )
                writer.add_scalar('review_loss', review_loss, step)
                logger.info(f"[Review] Step {step}: window_size={len(review_window)}, review_loss={review_loss:.4f}")

            # tensorboard
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('avg_loss', avg_loss / step, step)

            # log
            if step % config.log_freq == 0:
                end = int(time.time())
                logger.info(f"epoch:{str(epoch) + '/' + str(config.epochs)}, batch:{str(batch_idx + 1) + '/' + str(len(train_iter))}, step:{str(step) + '/' + str(total_step)}, cur_loss:{'{:.6f}'.format(loss)}, avg_loss:{'{:.6f}'.format(avg_loss/step)}, eta:{stats_time(start, end, step, total_step)}h")

        # Validation logic
        if val_iter is not None and epoch % config.val_freq == 0:
            logger.info(f"{'#' * 41} Validation Epoch {epoch} {'#' * 41}")
            val_metrics = validate(model, criterion, val_iter, config)
            logger.info(f"Validation Epoch {epoch}:")
            logger.info(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {val_metrics['precision']:.4f}")
            logger.info(f"  Recall:    {val_metrics['recall']:.4f}")
            logger.info(f"  F1:        {val_metrics['f1']:.4f}")

            # Log to tensorboard
            writer.add_scalar('val_accuracy', val_metrics['accuracy'], epoch)
            writer.add_scalar('val_precision', val_metrics['precision'], epoch)
            writer.add_scalar('val_recall', val_metrics['recall'], epoch)
            writer.add_scalar('val_f1', val_metrics['f1'], epoch)

            # Track best validation metric (do not save model)
            current_metric_value = val_metrics[config.metric_for_best]
            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_epoch = epoch
                logger.info(f"New best {config.metric_for_best}={best_metric_value:.4f} at epoch {epoch}")

    # Output overall metrics after training (only when use_llm=True)
    if llm_client is not None and (len(llm_preds_all) > 0 or len(model_preds_all) > 0):
        logger.info(f"{'#' * 41} Training Overall Metrics {'#' * 41}")

        # LLM overall metrics
        if len(llm_preds_all) > 0:
            llm_overall_metrics = compute_classification_metrics(llm_preds_all, llm_labels_all)
            logger.info(f"LLM Overall (High-Entropy, n={len(llm_preds_all)}):")
            logger.info(f"  Accuracy:  {llm_overall_metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {llm_overall_metrics['precision']:.4f}")
            logger.info(f"  Recall:    {llm_overall_metrics['recall']:.4f}")
            logger.info(f"  F1:        {llm_overall_metrics['f1']:.4f}")
        else:
            logger.info("LLM Overall: No high-entropy samples encountered")

        # Model overall metrics
        if len(model_preds_all) > 0:
            model_overall_metrics = compute_classification_metrics(model_preds_all, model_labels_all)
            logger.info(f"Model Overall (Low-Entropy, n={len(model_preds_all)}):")
            logger.info(f"  Accuracy:  {model_overall_metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {model_overall_metrics['precision']:.4f}")
            logger.info(f"  Recall:    {model_overall_metrics['recall']:.4f}")
            logger.info(f"  F1:        {model_overall_metrics['f1']:.4f}")
        else:
            logger.info("Model Overall: No low-entropy samples encountered")

        # System overall metrics (all samples)
        if len(llm_preds_all) > 0 and len(model_preds_all) > 0:
            all_preds = llm_preds_all + model_preds_all
            all_labels = llm_labels_all + model_labels_all
            overall_metrics = compute_classification_metrics(all_preds, all_labels)
            logger.info(f"System Overall (All samples, n={len(all_preds)}):")
            logger.info(f"  Accuracy:  {overall_metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {overall_metrics['precision']:.4f}")
            logger.info(f"  Recall:    {overall_metrics['recall']:.4f}")
            logger.info(f"  F1:        {overall_metrics['f1']:.4f}")
        elif len(llm_preds_all) > 0:
            logger.info(f"System Overall: Only LLM samples ({len(llm_preds_all)})")
        elif len(model_preds_all) > 0:
            logger.info(f"System Overall: Only Model samples ({len(model_preds_all)})")

        # Statistics
        total_samples = len(llm_preds_all) + len(model_preds_all)
        llm_ratio = len(llm_preds_all) / total_samples if total_samples > 0 else 0
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"LLM ratio: {llm_ratio:.2%}")

    # Save final model after training
    cur_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    weight_path = os.path.join(config.checkpoint_dir, "final_weight.pt")
    torch.save(model.model.state_dict(), cur_path)
    torch.save({'weight': model.weight}, weight_path)
    logger.info(f"Model saved to {cur_path}")

    # Output best model info
    if val_iter is not None:
        logger.info(f"{'#' * 41} Best Model {'#' * 41}")
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Best {config.metric_for_best}: {best_metric_value:.4f}")


if __name__ == '__main__':
    trainer()
