# reward functions

import torch
import numpy as np

import os

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

MODELS = {}

INFINIGRAM_CACHE_DIR = '/mnt/swordfish-pool2/horvitz/infinigram/v4_dolmasample_olmo'

def logmeanexp(scores):
    if not isinstance(scores, torch.Tensor):
        tensor_scores = torch.tensor(scores)
    else:
        tensor_scores = scores

    result = torch.logsumexp(tensor_scores, dim=-1) - np.log(tensor_scores.shape[-1])

    if not isinstance(scores, torch.Tensor):
        return result.tolist()
    else:
        return result

def _compute_roberta_score(
    *,
    model,
    tokenizer,
    texts,
    label_idx,
    device='cuda',
    delimiter='<|endoftext|>',
    just_first=True,
    batch_size=8,
    max_length=512,
):
    """
    Compute log mean probability of the label for each text.
    """
    # get individual texts
    all_texts = []
    original_indices = []

    for i, text in enumerate(texts):
        # currently batches within single generation
        split_text = [t for t in text.split(delimiter) if t.strip()]
        if just_first:
            split_text = split_text[:1]

        all_texts.extend(split_text)
        original_indices.extend([i] * len(split_text))

    # batch the input
    batched_input = []
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        batched_input.append(batch)

    # get scores
    all_scores = []
    for batch in batched_input:
        tokenized = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        outputs = model(**tokenized)
        # scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[:,label_idx].tolist()
        # use log softmax instead
        scores = torch.nn.functional.log_softmax(outputs.logits, dim=-1)[
            :, label_idx
        ].tolist()
        all_scores.extend(scores)

    # average the log scores
    unreduced_per_text_scores = [[] for _ in range(len(texts))]
    for i, score in zip(original_indices, all_scores):
        unreduced_per_text_scores[i].append(score)

    # avg_scores = [np.mean(scores) for scores in unreduced_per_text_scores]
   
    avg_scores = [logmeanexp(scores) for scores in unreduced_per_text_scores]
    return avg_scores, unreduced_per_text_scores


def sentiment_score(
    *,
    texts,
    label='positive',
    device='cuda',
    delimiter='<|endoftext|>',
    just_first=True,
    batch_size=8,
    max_length=512,
):
    '''Get sentiment score for a list of texts, each can have multiple documents separated by delimiter'''

    global MODELS

    if 'sentiment' not in MODELS:
        MODELS['sentiment'] = {
            'tokenizer': RobertaTokenizer.from_pretrained(
                'cardiffnlp/twitter-roberta-base-sentiment'
            ),
            'model': RobertaForSequenceClassification.from_pretrained(
                'cardiffnlp/twitter-roberta-base-sentiment'
            ),
        }
        MODELS['sentiment']['model'].eval()
        MODELS['sentiment']['model'].to(device)

    tokenizer = MODELS['sentiment']['tokenizer']
    model = MODELS['sentiment']['model']
    label_to_id = {'positive': 2, 'neutral': 1, 'negative': 0}

    label_idx = label_to_id[label]

    return _compute_roberta_score(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        label_idx=label_idx,
        device=device,
        delimiter=delimiter,
        just_first=just_first,
        batch_size=batch_size,
        max_length=max_length,
    )


def toxicity_score(
    *,
    texts,
    label='positive',
    device='cuda',
    delimiter='<|endoftext|>',
    just_first=True,
    batch_size=8,
    max_length=512,
    override_checkpoint=None,
):
    '''Get toxicity score for a list of texts, each can have multiple documents separated by delimiter'''

    global MODELS

    if override_checkpoint is not None:
        key = 'toxicity' + override_checkpoint
    else:
        key = 'toxicity'

    if key not in MODELS and override_checkpoint is None:
        MODELS[key] = {
            'tokenizer': RobertaTokenizer.from_pretrained(
                'SkolkovoInstitute/roberta_toxicity_classifier'
            ),
            'model': RobertaForSequenceClassification.from_pretrained(
                'SkolkovoInstitute/roberta_toxicity_classifier'
            ),
        }
        MODELS[key]['model'].eval()
        MODELS[key]['model'].to(device)
    elif key not in MODELS and override_checkpoint is not None:
        print(
            f'Overriding with best checkpoint from SSDLM Toxicity FT!!! {override_checkpoint}'
        )
        MODELS[key] = {
            'tokenizer': RobertaTokenizer.from_pretrained(
                'SkolkovoInstitute/roberta_toxicity_classifier'
            ),
            'model': RobertaForSequenceClassification.from_pretrained(
                override_checkpoint
            ),
        }
        MODELS[key]['model'] = RobertaForSequenceClassification.from_pretrained(
            override_checkpoint
        )
        MODELS[key]['model'].eval()
        MODELS[key]['model'].to(device)

    tokenizer = MODELS[key]['tokenizer']
    model = MODELS[key]['model']
    label_to_id = {'positive': 1, 'negative': 0}

    label_idx = label_to_id[label]

    return _compute_roberta_score(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        label_idx=label_idx,
        device=device,
        delimiter=delimiter,
        just_first=just_first,
        batch_size=batch_size,
        max_length=max_length,
    )


def formality_score(
    *,
    texts,
    label='formal',
    device='cuda',
    delimiter='<|endoftext|>',
    just_first=True,
    batch_size=8,
    max_length=512,
):
    global MODELS
    if 'formality' not in MODELS:
        MODELS['formality'] = {
            'tokenizer': RobertaTokenizer.from_pretrained(
                'cointegrated/roberta-base-formality'
            ),
            'model': RobertaForSequenceClassification.from_pretrained(
                'cointegrated/roberta-base-formality'
            ),
        }
        MODELS['formality']['model'].eval()
        MODELS['formality']['model'].to(device)

    tokenizer = MODELS['formality']['tokenizer']
    model = MODELS['formality']['model']

    label_to_id = {'formal': 1, 'informal': 0}
    label_idx = label_to_id[label]

    return _compute_roberta_score(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        label_idx=label_idx,
        device=device,
        delimiter=delimiter,
        just_first=just_first,
        batch_size=batch_size,
        max_length=max_length,
    )


def cola_score(
    *,
    texts,
    device='cuda',
    delimiter='<|endoftext|>',
    just_first=True,
    batch_size=32,
    max_length=512,
):
    '''Get CoLA score for a list of texts, each can have multiple documents separated by delimiter'''

    global MODELS

    if 'cola' not in MODELS:
        MODELS['cola'] = {
            'tokenizer': RobertaTokenizer.from_pretrained(
                'textattack/roberta-base-CoLA'
            ),
            'model': RobertaForSequenceClassification.from_pretrained(
                'textattack/roberta-base-CoLA'
            ),
        }
        MODELS['cola']['model'].eval()
        MODELS['cola']['model'].to(device)

    tokenizer = MODELS['cola']['tokenizer']
    model = MODELS['cola']['model']

    return _compute_roberta_score(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        label_idx=1,
        device=device,
        delimiter=delimiter,
        just_first=just_first,
        batch_size=batch_size,
        max_length=max_length,
    )


def gpt2_perp_score(
    *,
    texts,
    device='cuda',
    delimiter='<|endoftext|>',
    just_first=True,
    batch_size=8,
    add_start_token=True,
):
    '''Negative perplexity score for a list of texts, each can have multiple documents separated by delimiter'''
    global MODELS
    if 'gpt2' not in MODELS:
        MODELS['gpt2'] = {
            'tokenizer': AutoTokenizer.from_pretrained('gpt2'),
            'model': AutoModelForCausalLM.from_pretrained('gpt2'),
        }

        if MODELS['gpt2']['tokenizer'].pad_token is None:
            existing_special_tokens = list(
                MODELS['gpt2']['tokenizer'].special_tokens_map_extended.values()
            )
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."

            # assign one of the special tokens to also be the pad token
            MODELS['gpt2']['tokenizer'].add_special_tokens(
                {"pad_token": existing_special_tokens[0]}
            )

        MODELS['gpt2']['model'].eval()
        MODELS['gpt2']['model'].to(device)

    # perplexity = MODELS['perplexity']['metric']
    model = MODELS['gpt2']['model']
    tokenizer = MODELS['gpt2']['tokenizer']

    # get individual texts
    all_texts = []
    original_indices = []

    for i, text in enumerate(texts):
        # currently batches within single generation
        split_text = [t for t in text.split(delimiter) if t.strip()]
        if just_first:
            split_text = split_text[:1]

        all_texts.extend(split_text)
        original_indices.extend([i] * len(split_text))

    all_texts = [t.strip() for t in all_texts]

    # From
    # https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/ac4135177bfee71b1efd7bd3aff62e456e30aef9/perplexity.py

    if add_start_token:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = model.config.max_length - 1
    else:
        max_tokenized_len = model.config.max_length

    encodings = tokenizer(
        all_texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    perps = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for start_index in range(0, len(encoded_texts), batch_size):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
            ).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    attn_mask,
                ],
                dim=1,
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp2(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        # take log
        perplexity_batch = torch.log(perplexity_batch)  # just added!
        perps += perplexity_batch.tolist()

    # average the scores
    unreduced_per_text_scores = [[] for _ in range(len(texts))]
    for i, score in zip(original_indices, perps):
        unreduced_per_text_scores[i].append(-1 * score)

    # avg_scores = [np.mean(scores) for scores in unreduced_per_text_scores]
    avg_scores = [logmeanexp(scores) for scores in unreduced_per_text_scores]

    return avg_scores, unreduced_per_text_scores


def infinigram_perp_score(
    *,
    texts,
    device='cuda',
    delimiter='<|endoftext|>',
    just_first=True,
    batch_size=8,
    add_start_token=True,
    max_ngram=5,
    very_small_number=1e-10,
    max_num_samples=10,
):
    '''Negative perplexity score for a list of texts, each can have multiple documents separated by delimiter'''
    global MODELS

    if 'infinigram' not in MODELS:
        assert os.path.exists(
            INFINIGRAM_CACHE_DIR
        ), f"""Infinigram cache dir not found: {INFINIGRAM_CACHE_DIR}, see https://infini-gram.io/
Example cmd: `aws s3 cp --no-sign-request --recursive s3://infini-gram-lite/index/v4_dolmasample_olmo <LOCAL_INDEX_PATH>`
"""
        tokenizer = AutoTokenizer.from_pretrained(
            'allenai/OLMo-7B', trust_remote_code=True
        )
        MODELS['infinigram'] = {
            'tokenizer': tokenizer,
            'engine': InfiniGramEngine(
                index_dir=INFINIGRAM_CACHE_DIR,
                eos_token_id=tokenizer.eos_token_id,
            ),
        }

    engine = MODELS['infinigram']['engine']
    tokenizer = MODELS['infinigram']['tokenizer']

    # get individual texts
    all_texts = []
    original_indices = []

    for i, text in enumerate(texts):
        # currently batches within single generation
        split_text = [t for t in text.split(delimiter) if t.strip()]
        if just_first:
            split_text = split_text[:1]

        all_texts.extend(split_text)
        original_indices.extend([i] * len(split_text))

    all_texts = [t.strip() for t in all_texts]

    perps = []
    for text in all_texts:
        input_ids = tokenizer.encode(text)
        probs = []

        indices = list(range(len(input_ids)))
        if len(indices) > max_num_samples:
            indices = np.random.choice(indices, max_num_samples, replace=False)

        for i in indices:
            raw_prob = engine.prob(
                prompt_ids=input_ids[max(0, i - max_ngram) : i], cont_id=input_ids[i]
            )['prob']
            if raw_prob < very_small_number:
                # smooth
                raw_prob = very_small_number
            probs.append(raw_prob)

        # technically this is log perplexity
        perp = -1 * np.mean(np.log(probs))
        perps.append(perp)

    # # average the scores
    unreduced_per_text_scores = [[] for _ in range(len(texts))]
    for i, score in zip(original_indices, perps):
        unreduced_per_text_scores[i].append(-1 * score)

    # avg_scores = [np.mean(scores) for scores in unreduced_per_text_scores]
    avg_scores = [logmeanexp(scores) for scores in unreduced_per_text_scores]

    return avg_scores, unreduced_per_text_scores


if __name__ == '__main__':
    texts = [
        'I love this product! It is amazing! <|endoftext|> I hate this product! It is terrible!',
        'I love this product! It is amazing! <|endoftext|> I love this product! It is amazing!',
        'I hate this product! It is terrible! <|endoftext|> I hate this product! It is terrible!',
        'yo dude',
        'a he she a ate them',
    ]

    avg_scores, _ = sentiment_score(texts=texts, label='positive', just_first=False)
    print(avg_scores)
    assert avg_scores[2] < avg_scores[0] < avg_scores[1]

    avg_scores, _ = toxicity_score(texts=texts, label='positive')
    print(avg_scores)

    avg_scores, _ = formality_score(texts=texts, label='formal')
    print(avg_scores)
    assert avg_scores[0] > avg_scores[-2]

    avg_scores, _ = gpt2_perp_score(texts=texts)
    print(avg_scores)

    avg_scores, _ = cola_score(texts=texts)
    print(avg_scores)
    assert avg_scores[0] > avg_scores[-1]

    avg_scores, _ = infinigram_perp_score(texts=texts)
    print(avg_scores)
