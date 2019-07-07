import pandas as pd
import torch


def detokenize(pred, table, EOS_IDX):
    # print(pred)
    return "".join([table[tok] for tok in pred if tok > EOS_IDX])


def revtok(batch_pred, table, EOS_IDX=3):
    s = batch_pred.size()
    # print(pred)
    return [detokenize(batch_pred[:, col], table, EOS_IDX) for col in range(s[1])]


def evaluate_results(model, batch_iter, Q_TEXT, A_TEXT, path,name=""):
    model.eval()
    df = pd.DataFrame(columns=['Question', 'Answer', 'Output', 'Eval'])
    with torch.no_grad():
        for i, batch in enumerate(batch_iter):
            src = batch.Question
            trg = batch.Answer
            qus = revtok(src, Q_TEXT.vocab.itos, Q_TEXT.vocab.stoi['<eos>'])
            ans = revtok(trg, A_TEXT.vocab.itos, A_TEXT.vocab.stoi['<eos>'])
            # print(trg)
            output = model(src, trg, 0)  # turn off teacher forcing
            pred = output[1:].argmax(dim=2)
            pre = revtok(pred, A_TEXT.vocab.itos, A_TEXT.vocab.stoi['<eos>'])
            # print(qus)
            # print(ans)
            # print(pre)
            df = df.append(pd.DataFrame({'Question': qus, 'Answer': ans, 'Output': pre}), sort=False)

    df['Eval'] = (df['Output'] == df['Answer'])
    acc = df['Eval'].sum() / len(df['Eval'])
    df.to_csv(f'{path}/{name}_answered_{acc:.3f}.csv', index=False)
