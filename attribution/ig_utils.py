import torch
import numpy as np
from sty import fg, rs, Style, RgbFg

def get_ig_attributions(trunc_layer, model, tokenizer, query_tokens, para_tokens, label, 
        batch_size, device, attr_segment, begin_num_reps,
        max_reps, max_allowed_error, max_query_length, max_seq_length, debug):
    integrated_gradients, input_ids, baseline_prediction, prediction, error_percentage, num_reps = \
        _compute_ig(attr_segment=attr_segment, trunc_layer=trunc_layer, model=model, 
            tokenizer=tokenizer, query_tokens=query_tokens, 
            para_tokens=para_tokens, label=label, begin_num_reps=begin_num_reps, 
            batch_size=batch_size, device=device, 
            max_allowed_error=max_allowed_error, 
            max_reps=max_reps, max_query_length=max_query_length, 
            max_seq_length=max_seq_length, debug=debug)

    integrated_gradients = _project_attributions(tokenizer,
        input_ids, integrated_gradients)

    return integrated_gradients, error_percentage, baseline_prediction, prediction, num_reps


def visualize_token_attrs(tokens, attrs):
    """
      Visualize attributions for given set of tokens.
      Args:
      - tokens: An array of tokens
      - attrs: An array of attributions, of same size as 'tokens',
        with attrs[i] being the attribution to tokens[i]

      Returns:
      - visualization: An IPython.core.display.HTML object showing
        tokens color-coded based on strength of their attribution.
    """
    def get_color(attr):
        if attr > 0:
            g = int(128*attr) + 127
            b = 128 - int(64*attr)
            r = 128 - int(64*attr)
        else:
            g = 128 + int(64*attr)
            b = 128 + int(64*attr)
            r = int(-128*attr) + 127
        fg.color = Style(RgbFg(r, g, b))
        return fg.color

    # normalize attributions for visualization.
    bound = max(abs(np.max(attrs)), abs(np.min(attrs)))
    attrs = attrs/bound
    color_text = ""
    for i, tok in enumerate(tokens):
        color = get_color(attrs[i])
        color_text += " " + color + tok + fg.rs
    return color_text


def transform_input(tokenizer, query_tokens, para_tokens, label, device, max_query_length, max_seq_length):
    query_tokens = query_tokens[:max_query_length]
    max_para_tokens_len = max_seq_length - 3 - len(query_tokens)
    para_tokens = para_tokens[:max_para_tokens_len]
    input_tokens = ['[CLS]'] + query_tokens + ['[SEP]'] + para_tokens + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    token_type_ids = [0]*(2+len(query_tokens)) + [1]*(1+len(para_tokens))
    attention_mask = [1]*len(input_ids)
    assert len(input_ids) == len(token_type_ids) == len(attention_mask) <= max_seq_length
    '''
    if len(input_ids) < max_seq_length:
        pad_length = max_seq_length - len(input_ids)
        input_ids = input_ids + [0] * pad_length
        attention_mask = attention_mask + [0] * pad_length
        token_type_ids = token_type_ids + [0] * pad_length
    '''
    return (torch.LongTensor(input_ids).to(device), 
            torch.LongTensor(attention_mask).to(device), 
            torch.LongTensor(token_type_ids).to(device), 
            torch.LongTensor([label]).to(device))


def generate_baseline(attr_segment, tokenizer, query_tokens, para_tokens, label, device, max_query_length, max_seq_length):
    if attr_segment == "all" or attr_segment == "query":
        query_tokens = ["[PAD]"] * len(query_tokens)
    if attr_segment == "all" or attr_segment == "para":
        para_tokens = ["[PAD]"] * len(para_tokens)
    return transform_input(tokenizer, query_tokens, para_tokens, label, device, max_query_length, max_seq_length)


def _project_attributions(tokenizer, input_ids, attributions):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return {'outputs': [tokens, attributions.astype(np.float).tolist()]}


def _get_ig_error(integrated_gradients, baseline_prediction, prediction,
                  debug=False):
    sum_attributions = 0
    sum_attributions += np.sum(integrated_gradients)

    delta_prediction = prediction - baseline_prediction

    error_percentage = \
        100 * (delta_prediction - sum_attributions) / delta_prediction
    if debug:
        print(f'prediction is {prediction}')
        print(f'baseline_prediction is {baseline_prediction}')
        print(f'delta_prediction is {delta_prediction}')
        print(f'sum_attributions are {sum_attributions}')
        print(f'Error percentage is {error_percentage}')

    return error_percentage


def _get_scaled_inputs(input_val, baseline_val, scale_arr, batch_size):
    list_scaled_embeddings = []
    scaled_embeddings = \
        [baseline_val + scale * (input_val - baseline_val) for scale in scale_arr]

    while scaled_embeddings:
        list_scaled_embeddings.append(
            np.array(scaled_embeddings[:batch_size]))
        scaled_embeddings = scaled_embeddings[batch_size:]

    return list_scaled_embeddings


def _calculate_integral(ig):
    # We use np.average here since the width of each
    # step rectangle is 1/number of steps and the height is the gradient,
    # so summing the areas is equivalent to averaging the gradient values.

    ig = (ig[:-1] + ig[1:]) / 2.0  # trapezoidal rule

    integral = np.average(ig, axis=0)

    return integral


def _compute_ig(attr_segment, trunc_layer, model, tokenizer, query_tokens, para_tokens, label, begin_num_reps, batch_size, device, max_allowed_error, max_reps, max_query_length, max_seq_length, debug):
    for name, param in model.named_parameters():
        param.requires_grad = False
    model.eval()

    origin_input = transform_input(tokenizer, query_tokens, para_tokens, label, device, max_query_length, max_seq_length)
    input_ids, attention_mask, token_type_ids, _ = origin_input
    attention_mask = attention_mask.unsqueeze(0)
    token_type_ids = token_type_ids.unsqueeze(0)
    
    baseline_input_ids = generate_baseline(attr_segment, tokenizer, query_tokens, para_tokens, label, device)[0]
    
    input_ids
    batch_pre_input_ids = torch.cat(
        [input_ids.unsqueeze(0), baseline_input_ids.unsqueeze(0)], dim=0)
    _, hidden_states = model(trunc_layer=-1, 
                input_ids=batch_pre_input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, position_ids=None, head_mask=None, 
                inputs_embeds=None
    )
    # returned embeddings including word embeddings
    trunc_embeddings = hidden_states[trunc_layer].detach().cpu().numpy()
    origin_input_embds = trunc_embeddings[0]
    baseline_input_embds = trunc_embeddings[1]

    scale_arr = np.linspace(0, 1, begin_num_reps*batch_size)

    scaled_embeddings = _get_scaled_inputs(origin_input_embds, baseline_input_embds,
                                           scale_arr, batch_size)

    scores = []
    path_gradients = []
    baseline_prediction, prediction = None, None
    while True:
        for input_embds in scaled_embeddings:
            input_embds = torch.FloatTensor(input_embds).to(device)
            input_embds.requires_grad=True

            score_rep = model(trunc_layer=trunc_layer,
                input_ids=None, attention_mask=attention_mask, 
                token_type_ids=token_type_ids, position_ids=None, head_mask=None, 
                inputs_embeds=input_embds)[0][:, label]
            tp_sum = torch.sum(score_rep) # for backward
            tp_sum.backward()

            path_gradients_rep = input_embds.grad.detach().cpu().numpy()
            input_embds.requires_grad=False
            path_gradients.append(path_gradients_rep)
            scores.append(score_rep.detach().cpu().numpy())

        if baseline_prediction is None:
            baseline_prediction = scores[0][0]  # first score is the baseline prediction
        if prediction is None:
            prediction = scores[-1][-1]  # last score is the input prediction

        # integrating the gradients and multiplying with the difference of the
        # baseline and input.
        ig = np.concatenate(path_gradients, axis=0)
        
        arg_sort_idxes = np.argsort(scale_arr)
        ig = ig[arg_sort_idxes]

        integral = _calculate_integral(ig)
        integrated_gradients = (origin_input_embds - baseline_input_embds) * integral
        integrated_gradients = np.sum(integrated_gradients, axis=-1)
        error_percentage = \
            _get_ig_error(integrated_gradients, baseline_prediction, prediction, debug=debug)
        
        if abs(error_percentage) <= max_allowed_error or len(scaled_embeddings) >= max_reps:
            break
        sort_scale_arr = scale_arr[arg_sort_idxes]
        next_scale_arr = (sort_scale_arr[1:] + sort_scale_arr[:-1]) / 2.0
        scaled_embeddings = _get_scaled_inputs(origin_input_embds, baseline_input_embds,
                                           next_scale_arr, batch_size)
        scale_arr = np.concatenate((scale_arr, next_scale_arr), axis=0)


    input_ids = input_ids.detach().cpu().numpy()
    return (integrated_gradients, input_ids, 
        float(baseline_prediction), float(prediction), 
        float(error_percentage), len(scale_arr)) 
