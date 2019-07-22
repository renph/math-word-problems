import pandas as pd
import torch
from torchtext.data import TabularDataset, BucketIterator
import multiprocessing
from functools import partial
from itertools import takewhile


def detokenize(pred, table, EOS_IDX):
    # print(pred)
    # print(EOS_IDX)
    sen = takewhile(lambda x: x != EOS_IDX, pred)
    return "".join([table[tok] for tok in list(sen) if tok > EOS_IDX])


def next_col(batch_pred):
    batch_size = batch_pred.size()
    batch_pred = batch_pred.cpu()
    for col in range(batch_size[1]):
        yield batch_pred[:, col]


def revtok(batch_pred, table, EOS_IDX=3):
    # print(pred)
    pool = multiprocessing.Pool(4)
    # res = pool.map(lambda col: detokenize(batch_pred[:, col], table, EOS_IDX), range(s[1]))
    res = pool.map(partial(detokenize, table=table, EOS_IDX=EOS_IDX), next_col(batch_pred))
    # return [detokenize(batch_pred[:, col], table, EOS_IDX) for col in range(s[1])]
    return res


def evaluate_results(model, batch_iter, Q_TEXT, A_TEXT, path, name=""):
    model.eval()
    df = pd.DataFrame(columns=['Question', 'Answer', 'Output', 'Eval'])
    with torch.no_grad():
        for i, batch in enumerate(batch_iter):
            src = batch.Question
            trg = batch.Answer
            qus = revtok(src, Q_TEXT.vocab.itos, Q_TEXT.vocab.stoi['<eos>'])
            ans = revtok(trg, A_TEXT.vocab.itos, A_TEXT.vocab.stoi['<eos>'])
            # print(trg)
            output = model(src, trg)  # turn off teacher forcing
            pred = output[1:].argmax(dim=2)
            pre = revtok(pred, A_TEXT.vocab.itos, A_TEXT.vocab.stoi['<eos>'])
            # print(src[:, :5])
            # print(pred[:, :5])
            # print(qus[:5])
            # print(ans[:5])
            # print(pre[:5])
            df = df.append(pd.DataFrame({'Question': qus, 'Answer': ans, 'Output': pre}), sort=False)

    df['Eval'] = (df['Output'] == df['Answer'])
    acc = df['Eval'].sum() / len(df['Eval'])
    df.to_csv(f'{path}/{name}_answered_{acc:.3f}.csv', index=False)


def evaluate_dataset(dataset_path, data_fields, device, model, Q_TEXT, A_TEXT, save_path, name=""):
    tab_dataset = TabularDataset(path=f'{dataset_path}/all.csv', format='csv', fields=data_fields, skip_header=True)
    data_iter = BucketIterator(
        tab_dataset,
        batch_size=512,
        shuffle=False, sort=False,
        device=device
    )
    evaluate_results(model, data_iter, Q_TEXT, A_TEXT, save_path, name)
