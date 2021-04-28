import torch
from torch import optim
from torch import nn
from model import LSTM
from train import train, evaluate, predict
import copy
from config import Config
from preprocessing import PreProcessing
from os import path
import time

filename = ['training.csv', 'testing.csv']

if not (path.exists(filename[0]) and path.exists(filename[1])):
    print('Processing orginal files')
    pre = PreProcessing()
    pre.load_data()
    pre.process_train(filename[0])
    pre.process_test(filename[1])
else:
    print('Files exists!')

config = Config(filename)
config.print_para()
model = LSTM(config.vocab_size, config.embedding_dim, config.hidden_dim, config.label_size)
pretrained_embedding = config.TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embedding)

optimizer = optim.RMSprop(model.parameters(), lr=config.lr, momentum=0.9)

criterion = nn.CrossEntropyLoss()
print(vars(model))
model.cuda()
criterion.cuda()

best_valid_acc = 0.0
best_state_dict = copy.deepcopy(model.state_dict())
valid_list = []
train_list = []
print('Training started')
start_time = time.time()
for epoch in range(config.EPOCHS):
    epoch_start = time.time()
    train_loss, train_acc = train(model, config.train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, config.val_iter, criterion)
    valid_list.append(valid_acc)
    train_list.append(train_acc)
    epoch_end = time.time()
    print('Epoch {} | Train loss {:.3f}| Train acc {:.3f} | Valid loss {:.3f} | Valid acc {:.3f} | Time :{}'
          .format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc, epoch_end - epoch_start))

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_state_dict = copy.deepcopy(model.state_dict())
end_time = time.time()
print('total training time is :{}'.format(end_time - start_time))
print("Training finished")
print()
print('Best valid acc {:.3f}'.format(best_valid_acc))

# save model
model.load_state_dict(best_state_dict)
torch.save(model,
           './saved_model/model_embedding{}_hidden{}_epoch{}_lr{}_batch{}'.format(config.embedding_dim,
                                                                                  config.hidden_dim,
                                                                                  config.EPOCHS,
                                                                                  config.lr,
                                                                                  config.batch_size))
print('model saved!')

# test text predict and write file
test = []
vac = config.TEXT.vocab
print('start predicting')
for item in config.test_data.examples:
    string = item.test_text
    id = item.id
    ts = [vac[tok] for tok in string]
    test_tensor = torch.tensor([ts])
    label = predict(model, test_tensor)
    test.append([id, label])
print('start to write result file')
with open('./result/predicted.csv', 'w') as f:
    for item in test:
        f.write(item[0] + ',' + str(item[1]) + '\n')
