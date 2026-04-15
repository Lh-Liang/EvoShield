from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluation(eval_iter, model, criterion, device) -> dict:
    all_labels = []
    all_predictions = []

    for i, batch in enumerate(eval_iter):
        model.eval()
        input_ids, token_type_ids, attention_mask, mlm_labels, labels, _ = [
            w.to(device) if hasattr(w, "to") else w for w in batch
        ]

        logit, weight= model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        _, predictions = criterion(logit, mlm_labels, labels, weight)

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.cpu().detach().numpy().reshape(-1).tolist())

    return {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'macro_precision': precision_score(all_labels, all_predictions, average='macro', zero_division=0),
        'macro_recall': recall_score(all_labels, all_predictions, average='macro', zero_division=0),
        'macro_f1': f1_score(all_labels, all_predictions, average='macro', zero_division=0),
    }
