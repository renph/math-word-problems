import copy
import math
import random
import time
import argparse

import torch
from torch import nn, optim
from torchtext.data import Field, BucketIterator, TabularDataset

from model.seq2seq import Encoder, Decoder, Seq2Seq
from utils.config import Config
from utils import count_parameters, fix_seed, epoch_time

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='',
                    help='Path to config file')

args = parser.parse_args(args=['--config', 'config.txt'])
if args.config:
    args = Config(args.config)
if args.show:
    from utils import enable_tqdm as tqdm
else:
    from utils import disable_tqdm as tqdm

fix_seed(args.seed)

# def tokenize(sentence):
#     return [tok for tok in sentence]

Q_TEXT = Field(tokenize=lambda sen: list(sen), init_token="<sos>", eos_token="<eos>")
A_TEXT = Field(tokenize=lambda sen: list(sen), init_token="<sos>", eos_token="<eos>")

# associate the text in the 'Question' column with the Q_TEXT field,
# and 'Answer' with A_TEXT field
data_fields = [('Question', Q_TEXT), ('Answer', A_TEXT)]

# train, val = TabularDataset.splits(path=PATH, train='train.csv', validation='val.csv', format='csv',
#                                    fields=data_fields, skip_header=True)
tab_dataset = TabularDataset(path=f'{args.path}/all.csv', format='csv', fields=data_fields, skip_header=True)
train, val, test = tab_dataset.split(split_ratio=[0.5, 0.2, 0.3], random_state=random.getstate())

Q_TEXT.build_vocab(train)
A_TEXT.build_vocab(train)
print('Question Tokenize')
print(list(Q_TEXT.vocab.stoi.items()))
print('Answer Tokenize')
print(list(A_TEXT.vocab.stoi.items()))
# print(list(A_TEXT.vocab.itos))

INPUT_DIM = len(Q_TEXT.vocab)
OUTPUT_DIM = len(A_TEXT.vocab)

# BATCH_SIZE = 512
# ENC_EMB_DIM = 256  # 256
# DEC_EMB_DIM = 256  # 256
# HID_DIM = 512  # 512
# N_LAYERS = 2
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'using device {device}')

train_iter, val_iter, test_iter = BucketIterator.splits(
    (train, val, test),
    batch_size=args.batch_size,
    shuffle=True, sort=False,
    device=device
)

# build model
enc = Encoder(INPUT_DIM, args.enc_emb_dim, args.hid_dim, args.n_layers, args.enc_dropout).to(device)
dec = Decoder(OUTPUT_DIM, args.dec_emb_dim, args.hid_dim, args.n_layers, args.dec_dropout).to(device)
model = Seq2Seq(enc, dec, device).to(device)

print(model)
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=1e-3)
PAD_IDX = A_TEXT.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        src = batch.Question
        trg = batch.Answer

        optimizer.zero_grad()
        output = model(src, trg)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        # print(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def accuracy(pred, target, EOS_IDX=3):
    s = pred.size()
    finished = torch.zeros(size=(s[1],), dtype=torch.uint8, device=device)
    correct = torch.ones_like(finished, dtype=torch.uint8, device=device)
    for i in range(s[0]):
        correct *= ((pred[i] == target[i]) + finished) > 0
        finished += (target[i] == EOS_IDX)
    return correct.sum().item()


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.Question
            trg = batch.Answer

            output = model(src, trg, 0)  # turn off teacher forcing
            pred = output[1:].argmax(dim=2)

            epoch_acc += accuracy(pred, trg[1:], EOS_IDX=A_TEXT.vocab.stoi['<eos>'])
            total += len(batch)
            # print(output.shape)
            # print()
            # print(trg.shape)
            # print()
            # raise Exception('stop')

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator), epoch_acc / total


best_valid_loss = float('inf')
best_valid_acc = 0
best_model = {}
try:
    for epoch in range(args.n_epochs):
        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, criterion, args.clip)
        valid_loss, valid_acc = evaluate(model, val_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_acc > best_valid_acc and valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} |  Val. ACC: {valid_acc:.3f}')
        if valid_acc > 0.93 or train_loss < 0.005:  # early stop
            break
except Exception as e:
    print(e)
finally:
    print(f'Best Val. Loss: {best_valid_loss:.3f} |  Val. ACC: {best_valid_acc:.3f}')
    torch.save(best_model, f'{args.path}/best-{best_valid_acc:.3f}-{args.hid_dim}-{args.seed}.pt')
    model.load_state_dict(best_model)
    _, acc = evaluate(model, test_iter, criterion)
    print(f'Test ACC: {acc:.3f}')
    model.eval()

    from utils.postprocess import evaluate_results

    evaluate_results(model, test_iter, Q_TEXT, A_TEXT, f'{args.path}', 'test')
