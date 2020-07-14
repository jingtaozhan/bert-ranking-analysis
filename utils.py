def is_num(s):
    return all('0'<=c<='9' for c in s)


def get_period_idxes(tokens):
    period_idxes = [idx for idx, token in enumerate(tokens) 
        if token =="." and (
        idx==0 or not is_num(tokens[idx-1]) or 
        idx==len(tokens)-1 or not is_num(tokens[idx+1]))
    ]
    return period_idxes