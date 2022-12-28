import numpy as np, math,  sys
sys.path.append('../')
from collections import defaultdict as ddict
from tqdm import tqdm
from copy import deepcopy
import torch

word_idx_map, word_features, adj = [None] * 3

def check_monotone(baseline, input, candidate, ret_type = 'vec'):
    
    '''
    
    Check that candidate is monotonic between baseline and input.
    ret_type consist of bool, count, vec.
    Count case is used to Max Count anchor search method.
    Vec case is used to Greedy anchor search method.
    Bool case is used to check all interpoloation point are monotonic between baseline & input. 
    
    '''
    
    increasing_dims		= baseline > input 	# dims where baseline > input
    decreasing_dims		= baseline < input 	# dims where baseline < input
    equal_dims			= baseline == input 	# dims where baseline == input

    # check candidate cond.
    
    vec3_greater_vec1	= candidate >= baseline
    vec3_greater_vec2	= candidate >= input
    vec3_lesser_vec1	= candidate <= baseline
    vec3_lesser_vec2	= candidate <= input
    vec3_equal_vec1		= candidate == baseline
    vec3_equal_vec2		= candidate == input
 
    
    # check monotonize
    # First Case, Baseline > Input -> Baseline >= Candidate >= Input
    # Second Case, Baseline < Input -> Baseline <= Candidate <= Input
    # Third Case, Baseline == Input -> Baseline == Candidate == Input
    
    monotone = (increasing_dims * vec3_lesser_vec1 * vec3_greater_vec2 + decreasing_dims * vec3_greater_vec1 * vec3_lesser_vec2 + equal_dims * vec3_equal_vec1 * vec3_equal_vec2)
    
    # Return each case.
     
    if ret_type == 'bool':
        return monotone.sum() == baseline.shape[0]
    elif ret_type == 'count':
        return monotone.sum()
    elif ret_type == 'vec':
        return monotone
    
def create_monotonic_vec(baseline, input, candidate, interpolation_step):
    
    '''
    
    Create monotonic vector.
    In detail, change from candidate vector to monotonic vector.
    
    '''
 
    # Check 
    monotone_dims = check_monotone(baseline, input, candidate, ret_type='vec')
    non_monotone_dims = ~monotone_dims
    
    # If number of non_monotone_dims equal 0, then all dims are monotone. 
    if non_monotone_dims.sum() == 0:
        return candidate

    # make anchor monotonic
    monotone_vec = deepcopy(candidate)
    monotone_vec[non_monotone_dims] = input[non_monotone_dims] - (1.0 / interpolation_step) * (input[non_monotone_dims] - baseline[non_monotone_dims])

    return monotone_vec

def distance(A, B):
    
    '''
	return eucledian distance between two points
    '''
    
    return np.sqrt(np.sum((A - B) ** 2))


def find_next_wrd(wrd_idx, ref_idx, word_path, strategy='greedy', steps=30):
    
    '''
    
    Given word_index & reference_index, find next anchor point.
    wrd_idx : word_index (input).
    ref_idx : reference_index (baseline).
    word_path : word path up to the previous point.
    
    '''
    
    global adj, word_features


    if wrd_idx == ref_idx:
        # If (for some reason) we do select the ref_idx as the previous anchor word, then all further anchor words should be ref_idx
        return ref_idx

    anchor_map = ddict(list)
    cx = adj[wrd_idx].tocoo()

    for candidate,_ in zip(cx.col, cx.data):
        # Should not consider the anchor word to be the ref_idx [baseline] unless forced to.
        if candidate == ref_idx:
            continue

        # Anchor Search Algorithm.

        if strategy == 'greedy':
            # calculate the distance of the monotonized vec from the anchor point
            monotonic_vec	= create_monotonic_vec(word_features[ref_idx], word_features[wrd_idx], word_features[candidate], steps)
            anchor_map[candidate]	= [distance(word_features[candidate], monotonic_vec)]

        elif strategy == 'maxcount':
            # count the number of non-monotonic dimensions (10000 is an arbitrarily high and is a hack to be agnostic of word_features dimension)
            non_mono_count	= 10000 - check_monotone(word_features[ref_idx], word_features[wrd_idx], word_features[candidate], ret_type='count')
            anchor_map[candidate]	= [non_mono_count]

        else:
            raise NotImplementedError

    if len(anchor_map) == 0:
        return ref_idx

    sorted_dist_map = {k: v for k, v in sorted(anchor_map.items(), key=lambda item: item[1][0])}

	# remove words that are already selected in the path
    for key in word_path:
        sorted_dist_map.pop(key, None)

    if len(sorted_dist_map) == 0:
        return ref_idx

    # return the top key
    return next(iter(sorted_dist_map))


def find_word_path(wrd_idx, ref_idx, steps=30, strategy='greedy'):
    
    '''
    
    Given, word_index & reference_index, find all anchor points.
    All points in this path are not necessarily interpolation points.
    After finding word path, monotonize each point.
    
    
    '''
    
    global word_idx_map

    # if wrd_idx is CLS or SEP then just copy that and return
    if ('[CLS]' in word_idx_map and wrd_idx == word_idx_map['[CLS]']) or ('[SEP]' in word_idx_map and wrd_idx == word_idx_map['[SEP]']):
        return [wrd_idx] * (steps+1)

    word_path	= [wrd_idx]
    last_idx	= wrd_idx
    for _ in range(steps):
        next_idx = find_next_wrd(last_idx, ref_idx, word_path, strategy=strategy, steps=steps)
        word_path.append(next_idx)
        last_idx = next_idx
    return word_path


def make_monotonic_path(word_path_ids, ref_idx, steps=30):
    
    '''
    
    Given word path & reference index, modify each point so that monotone. 
    Return value is final path of interpolation points.
    
    '''
    
    global word_features
    monotonic_embs = [word_features[word_path_ids[0]]]
    vec1 = word_features[ref_idx]

    for idx in range(len(word_path_ids)-1):
        vec2 = monotonic_embs[-1]
        vec3 = word_features[word_path_ids[idx+1]]
        vec4 = create_monotonic_vec(vec1, vec2, vec3, steps)
        monotonic_embs.append(vec4)
    monotonic_embs.append(vec1)

    # reverse the list so that baseline is the first and input word is the last
    monotonic_embs.reverse()

    final_embs = monotonic_embs

    # verify monotonicity
    check = True
    for i in range(len(final_embs)-1):
        check *= check_monotone(final_embs[-1], final_embs[i], final_embs[i+1], ret_type='bool')
    assert check

    return final_embs


def making_interpolation_path(input_ids, ref_input_ids, device, auxiliary_data, steps=30, strategy='greedy'):
 
    '''	
    generates the paths required by DIG
    '''
 
    global word_idx_map, word_features, adj
    word_idx_map, word_features, adj = auxiliary_data

    all_path_embs	= []
    for idx in tqdm(range(len(input_ids))):
        word_path = find_word_path(input_ids[idx], ref_input_ids[idx], steps=steps, strategy=strategy)
        monotonic_embs = make_monotonic_path(word_path, ref_input_ids[idx], steps=steps)
        all_path_embs.append(monotonic_embs)
    all_path_embs = torch.tensor(np.stack(all_path_embs, axis=1), dtype=torch.float, device=device, requires_grad=True)

    return all_path_embs