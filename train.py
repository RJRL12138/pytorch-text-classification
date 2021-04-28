import torch


def train(model, iterator, optimizer, criterion):
    total_loss, total_correct, total_prediction = 0.0, 0.0, 0.0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        out_logits = model(batch.text.cuda())
        predictions = torch.max(out_logits, dim=-1)[1]
        loss = criterion(out_logits.squeeze(1), batch.label.cuda())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += torch.eq(predictions, batch.label.cuda()).sum().item()
        total_prediction += batch.label.size(0)
    epoch_loss = total_loss / len(iterator)
    epoch_acc = total_correct / total_prediction
    return epoch_loss, epoch_acc


def evaluate(model, iterator, criterion):
    total_loss, total_correct, total_prediction = 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            eva_logits = model(batch.text.cuda())
            predictions = torch.max(eva_logits, dim=-1)[1]
            loss = criterion(eva_logits.squeeze(1), batch.label.cuda())

            total_loss += loss.item()
            total_correct += torch.eq(predictions, batch.label.cuda()).sum().item()
            total_prediction += batch.label.size(0)
    return total_loss / len(iterator), total_correct / total_prediction


def predict(model, text):
    model.eval()
    with torch.no_grad():
        pred = model(text.cuda())
        out = torch.max(pred, dim=-1)[1]
        return out.cpu().numpy()[0]
