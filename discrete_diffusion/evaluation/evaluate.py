# Based on https://github.com/xhan77/ssd-lm
'''

Example usage:

python evaluate.py \
--generations_file '[path_to_gen].jsonl' \
--metrics ppl#gpt2-xl,cola,dist-n,toxic,toxic_ext \
--output_file '[path_to_gen]_eval.txt'


'''
import pandas as pd
from pathlib import Path
import os
import numpy as np

import click
import torch

import os
import logging


from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

logger = logging.getLogger(__name__)


def conditional_perplexity(
    generations_df, model, tokenizer, device='cuda', write_file=None
):
    perplexities = []
    goodperplexities = []
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0
    if write_file is not None:
        fout = open(write_file, "w")

    # for every prompt
    for i, row in tqdm(
        generations_df.iterrows(),
        total=len(generations_df.index),
        desc='Evaluating PPL',
    ):
        # prompt_input_ids = torch.LongTensor([row.prompt['tokens']]).to(device)
        prompt = row.context_string
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        if not (
            prompt_input_ids.shape[1] == 1
            and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id
        ):  # this means unconditional, prompt is BOS token (verify)
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (
                prompt_input_ids.shape[1] - 1
            )
            # print("in")
        else:
            prompt_loss = 0
            # print("out")
        # for every generation conditioned on the prompt
        generations = row.string
        for gen in generations:
            full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(
                device
            )
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (
                full_input_ids.shape[1] - 1
            )
            loss = (full_loss - prompt_loss) / (
                full_input_ids.shape[1] - prompt_input_ids.shape[1]
            )

            ppl = np.exp(loss.item())
            if ppl < 100:  # for sanity
                goodperplexities.append(ppl)
                # perplexities.append(ppl)
                g += 1

            if ppl < 1e4:
                perplexities.append(ppl)
            else:
                print("ppl values are weirldly large. Check for errors")
                print(f"\n########\n{gen}\n########\n")

            total_nll += (full_loss - prompt_loss).item()
            total_tokens += full_input_ids.shape[1] - prompt_input_ids.shape[1]
            # print(full_input_ids[0], prompt_input_ids[0])
            # print(full_loss, prompt_loss)
            # input()
            if write_file is not None:
                fout.write(
                    f"{ppl}, {(full_loss - prompt_loss).item()}, {(full_input_ids.shape[1] - prompt_input_ids.shape[1])}\n"
                )

    return np.nanmean(perplexities), np.exp(total_nll / total_tokens)


def fluency_classify(generations_df):
    ctr_model_name = 'textattack/roberta-base-CoLA'
    ctr_label_idx = 1
    tokenizer = AutoTokenizer.from_pretrained(ctr_model_name)
    config = AutoConfig.from_pretrained(ctr_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')

    machine_text = []
    for i, row in generations_df.iterrows():
        prompt = row['context_string']
        for output in row['string']:
            machine_text.append(f'{prompt}{output}')

    acc_list = []
    for text in machine_text:
        encoded_input = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')
        output = model(**encoded_input)
        argmax_label_list = output.logits.argmax(dim=-1).view(-1).tolist()
        for _l in argmax_label_list:
            if _l == ctr_label_idx:
                acc_list.append(1)
            else:
                acc_list.append(0)
    acc = sum(acc_list) / len(acc_list)
    return acc


def distinctness(generations_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(
        generations_df.iterrows(),
        total=len(generations_df.index),
        desc='Evaluating dist-n',
    ):
        generations = row['string']
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            # o = [str(tok) for tok in gen]
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


def compute_toxic(generations_df, ctr_label_idx):  #
    ctr_model_name = 'SkolkovoInstitute/roberta_toxicity_classifier'
    # if ctr_label_idx == 2:
    #     ctr_label_idx = 1 # cardiffnlp/twitter-roberta-base-sentiment's label idx 2 is positive sentiment (1 in other models)

    tokenizer = AutoTokenizer.from_pretrained(ctr_model_name)
    config = AutoConfig.from_pretrained(ctr_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')

    machine_text = []
    for i, row in generations_df.iterrows():
        prompt = row['context_string']
        for output in row['string']:
            machine_text.append(f'{prompt}{output}')

    acc_list = []
    for text in machine_text:
        encoded_input = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')
        output = model(**encoded_input)
        argmax_label_list = output.logits.argmax(dim=-1).view(-1).tolist()
        for _l in argmax_label_list:
            if _l == ctr_label_idx:
                acc_list.append(1)
            else:
                acc_list.append(0)
    acc = sum(acc_list) / len(acc_list)
    return acc


def compute_toxic_ext(generations_df, ctr_label_idx):  #
    ctr_model_name = 'textdetox/xlmr-large-toxicity-classifier'

    tokenizer = AutoTokenizer.from_pretrained(ctr_model_name)
    config = AutoConfig.from_pretrained(ctr_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')

    machine_text = []
    for i, row in generations_df.iterrows():
        prompt = row['context_string']
        for output in row['string']:
            machine_text.append(f'{prompt}{output}')

    acc_list = []
    for text in machine_text:
        encoded_input = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')
        output = model(**encoded_input)
        argmax_label_list = output.logits.argmax(dim=-1).view(-1).tolist()
        for _l in argmax_label_list:
            if _l == ctr_label_idx:
                acc_list.append(1)
            else:
                acc_list.append(0)
    acc = sum(acc_list) / len(acc_list)
    return acc


@click.command()
@click.option(
    '--generations_file',
    required=True,
    type=str,
    help='a jsonl file with generations and attribute scores',
)
@click.option(
    '--output_file', required=True, type=str, help='filename to write the results to'
)
@click.option(
    '--metrics',
    required=True,
    type=str,
    help='which metrics to compute, write comma separeted, ppl-mid,ppl-big,cola,self-bleu,zipf,repetition,dist-n',
)
@click.option('--extra', required=False, type=str, help='extra params')
def main(generations_file, output_file, metrics, extra):
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    generations_df = pd.read_json(generations_file, lines=True)

    metricset = set(metrics.strip().split(","))  # cannot use lower here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### calculate quality metrics

    # Fluency
    fo = open(output_dir / output_file, 'w')  # creating the file
    fo.close()

    # print(metrics)
    if "ppl" in metrics:
        for metric in metricset:
            if "ppl" in metric:
                eval_modelname = metric.split("#")[1]
                print(f'computing {eval_modelname} ppl')
                if 'llama3' in eval_modelname:
                    LLAMA_TOKEN = os.environ['LLAMA_TOKEN']
                    model_name = "meta-llama/Meta-Llama-3-8B"
                    print(f"Loading {model_name}")
                    eval_model = AutoModelForCausalLM.from_pretrained(
                        model_name, use_auth_token=LLAMA_TOKEN
                    ).to(device)
                    eval_tokenizer = AutoTokenizer.from_pretrained(
                        model_name, use_auth_token=LLAMA_TOKEN
                    )
                else:
                    eval_model = AutoModelForCausalLM.from_pretrained(
                        eval_modelname
                    ).to(device)
                    eval_tokenizer = AutoTokenizer.from_pretrained(eval_modelname)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    ppl, total_ppl = conditional_perplexity(
                        generations_df,
                        eval_model,
                        eval_tokenizer,
                        device=device,
                        write_file=output_dir
                        / (output_file + ".ppl-" + eval_modelname.replace("/", "-")),
                    )

                # write output results
                with open(output_dir / output_file, 'a') as fo:
                    fo.write(
                        f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n'
                    )
                    print(
                        f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n'
                    )

                del eval_model

    # cola
    if "cola" in metricset:
        print("computing fluency (cola)")
        # cola_accuracy = fluency_classify(generations_df, output_file=output_dir / (output_file+".cola"))
        cola_accuracy = fluency_classify(generations_df)

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'cola acceptability accuracy = {cola_accuracy}\n')
            print(cola_accuracy)

    ### calculate diversity
    # dist-n
    if "dist-n" in metricset:
        dist1, dist2, dist3 = distinctness(generations_df)

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            for i, dist_n in enumerate([dist1, dist2, dist3]):
                fo.write(f'dist-{i+1} = {dist_n}\n')
                print(f'dist-{i+1} = {dist_n}')

    if "toxic" in metricset:
        acc = compute_toxic(generations_df, 1)
        #if 'negative' in generations_file:
        #    acc = compute_toxic(generations_df, 0)
        #elif 'positive' in generations_file:
        #    acc = compute_toxic(generations_df, 1)
        #else:
        #    raise ValueError("check ctridx")

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'toxic acc = {acc}\n')
            print(f'toxic acc = {acc}')

    if "toxic_ext" in metricset:
        acc = compute_toxic_ext(generations_df, 1)
        #if 'negative' in generations_file:
        #    acc = compute_toxic_ext(generations_df, 0)
        #elif 'positive' in generations_file:
        #    acc = compute_toxic_ext(generations_df, 1)
        #else:
        #    raise ValueError("check ctridx")

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'toxic_ext acc = {acc}\n')
            print(f'toxic_ext acc = {acc}')


if __name__ == '__main__':
    main()
