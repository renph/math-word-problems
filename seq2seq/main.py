import copy
import math
import random
import time

import pandas as pd
import torch
from torch import nn, optim
from torchtext.data import Field, BucketIterator, TabularDataset

from seq2seq import Encoder, Decoder, Seq2Seq
from utils import MathLang

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# math_lang = MathLang('math-10')
#
# df = pd.read_csv("../tmp/train_arith_10/train.csv")
# for q in df["Question"]:
#     math_lang.addSentence(q)
# print(math_lang.word2index)
# print(math_lang.index2word)

# PATH = '../tmp/train_arith_10_1111'
# PATH = '../tmp/train_arith_3_50_1111'
PATH = '../tmp/train_arith_3_100_4111'
# PATH = '../tmp/train_mixedarith_3_100_1111'

# def tokenize(sentence):
#     return [tok for tok in sentence]


Q_TEXT = Field(tokenize=lambda sen: list(sen), init_token="<sos>", eos_token="<eos>")
A_TEXT = Field(tokenize=lambda sen: list(sen), init_token="<sos>", eos_token="<eos>")

# associate the text in the 'English' column with the EN_TEXT field, # and 'French' with FR_TEXT
data_fields = [('Question', Q_TEXT), ('Answer', A_TEXT)]
train, val = TabularDataset.splits(path=PATH, train='train.csv', validation='val.csv', format='csv',
                                   fields=data_fields, skip_header=True)
Q_TEXT.build_vocab(train)
A_TEXT.build_vocab(train)
print('Question Tokenize')
print(list(Q_TEXT.vocab.stoi.items()))
print('Answer Tokenize')
print(list(A_TEXT.vocab.stoi.items()))
# print(list(A_TEXT.vocab.itos))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'using {device}')

BATCH_SIZE = 512
INPUT_DIM = len(Q_TEXT.vocab)
OUTPUT_DIM = len(A_TEXT.vocab)
ENC_EMB_DIM = 256  # 256
DEC_EMB_DIM = 256  # 256
HID_DIM = 512  # 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

train_iter, val_iter = BucketIterator.splits(
    (train, val),
    batch_size=BATCH_SIZE,
    shuffle=True, sort=False,
    device=device
)

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)
model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)

print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=1e-3)
PAD_IDX = A_TEXT.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10000
CLIP = 1

best_valid_loss = float('inf')
best_valid_acc = 0
best_model = {}
for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss, valid_acc = evaluate(model, val_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss and valid_acc > best_valid_acc:
        best_valid_loss = valid_loss
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model.state_dict())

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} |  Val. ACC: {valid_acc:.3f}')
    if valid_acc > 0.75:  # early stop
        break
print(f'Best Val. Loss: {best_valid_loss:.3f} |  Val. ACC: {best_valid_acc:.3f}')
print()
torch.save(best_model, f'{PATH}/best-{best_valid_acc:.3f}-{HID_DIM}-{SEED}.pt')


def detokenize(pred, table, EOS_IDX):
    # print(pred)
    return "".join([table[tok] for tok in pred if tok > EOS_IDX])


def revtok(batch_pred, table, EOS_IDX=3):
    # from multiprocessing import Process, Manager
    # pool = []
    # manager = Manager()
    # return_dict = manager.dict()
    # return_dict['tab'] = table

    s = batch_pred.size()
    # print(pred)
    return [detokenize(batch_pred[:, col], table, EOS_IDX) for col in range(s[1])]


if __name__ == '__main__':
    df = pd.DataFrame(columns=['Question', 'Answer', 'Output', 'Eval'])
    model.load_state_dict(best_model)
    _, acc = evaluate(model, val_iter, criterion)
    print(acc)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_iter):
            src = batch.Question
            trg = batch.Answer
            qus = revtok(src, Q_TEXT.vocab.itos, Q_TEXT.vocab.stoi['<eos>'])
            print(qus)
            ans = revtok(trg, A_TEXT.vocab.itos, A_TEXT.vocab.stoi['<eos>'])
            print(ans)
            # print(trg)
            output = model(src, trg, 0)  # turn off teacher forcing
            pred = output[1:].argmax(dim=2)
            pre = revtok(pred, A_TEXT.vocab.itos, A_TEXT.vocab.stoi['<eos>'])
            print(pre)

            df = df.append(pd.DataFrame({'Question': qus, 'Answer': ans, 'Output': pre}), sort=False)

    df['Eval'] = (df['Output'] == df['Answer'])
    acc = df['Eval'].sum() / len(df['Eval'])
    df.to_csv(f'{PATH}/val_answered_{acc:.3f}.csv', index=False)
