# unary operations
# identity, square, 2, -, -2, inv, centralize
IDENTITY = 0
SQUARE = 1
DOUBLE = 2
NEGATIVE = 3
# UNARY_INVALID = 4 # restrict unary version nn = 4
CENTRALIZE = 4
ORTHOGONIZE = 5
INVERSE = 6
UNARY_INVALID = 7

# binary operations
# add, minus, element-wise multiply, cross product, cartesian product, matrix-vector product
ADD = 0
MINUS = 1
MULTIPLY = 2
# BINARY_INVALID = 3 # to 3
CROSS_PRODUCT = 3
CARTESIAN_PRODUCT = 4
MATRIX_VECTOR_PRODUCT = 5
BINARY_INVALID = 6

# grouping operation
# sum, mean, max, svd
SUM = 0
MEAN = 1
MAX = 2
SVD = 3
GROUP_INVALID = 4


def convert_unary_oper_to_str(uop, oper):
    oper_str = oper
    if uop == 0:
        ans = oper_str
    elif uop == 1:
        ans = "(" + oper_str + ")" + "^2"
    elif uop == 2:
        ans = "(2*" + oper_str + ")"
    elif uop == 3:
        ans = "(-" + oper_str + ")"
    elif uop == INVERSE:
        ans = "inv(" + oper_str + ")"
    elif uop == 4:
        ans = "centralize(" + oper_str + ")"
    elif uop == 5:
        ans = "orth(" + oper_str + ")"
    else:
        raise ValueError(f"Unrecognized uop: {uop}")
    return ans


def convert_group_oper_to_str(gop, oper):
    if gop == 0:
        ans = "sum(" + oper + ")"
    elif gop == 1:
        ans = "mean(" + oper + ")"
    elif gop == 2:
        ans = "max(" + oper + ")"
    elif gop == 3:
        ans = "svd(" + oper + ")"
    else:
        raise ValueError(f"Unrecognized gop: {gop}")
    return ans


# add, minus, element-wise multiply, cross product, cartesian product, matrix-vector product
def convert_binary_oper_to_str(bop, lft_oper, rgt_oper):
    if bop == 0:
        ans = lft_oper + "+" + rgt_oper
    elif bop == 1:
        ans = lft_oper + "-" + rgt_oper
    elif bop == 2:
        ans = lft_oper + "*" + rgt_oper
    elif bop == 3:
        ans = "cross(" + lft_oper + "," + rgt_oper + ")"
    elif bop == 4:
        ans = "cartesian(" + lft_oper + "," + rgt_oper + ")"
    elif bop == 5:
        ans = "matmul(" + lft_oper + "," + rgt_oper + ")"
    else:
        raise ValueError(f"Unrecognized bop: {bop}")
    return ans

def convert_operants_idx_to_str(operant_idx):
    if operant_idx == 0:
        return "P"
    elif operant_idx == 1:
        return "N"
    elif operant_idx == 2:
        return "(P*N)"
    elif operant_idx == 3:
        return "(P+N)"
    elif operant_idx == 4:
        return "(P-N)"
    elif operant_idx == 5:
        return "(N-P)"
    elif operant_idx == 6:
        return "cross(P,N)"
    else:
        raise ValueError(f"Unrecognized operant idx: {operant_idx}.")


def conver_oper_dict_to_readable_dict(oper_dict):
    if 'oper' in oper_dict:
        cur_oper = oper_dict['oper']
        # oper_str = "P" if cur_oper == 0 else "N"
        oper_str = convert_operants_idx_to_str(cur_oper)
        uop = oper_dict['uop']
        ans = convert_unary_oper_to_str(uop, oper_str)
    elif 'chd' in oper_dict:
        uop = oper_dict['uop']
        gop = oper_dict['gop']
        chd_ans = conver_oper_dict_to_readable_dict(oper_dict['chd'])
        ans = convert_group_oper_to_str(gop, chd_ans)
        ans = convert_unary_oper_to_str(uop, ans)

    elif 'lft_chd' in oper_dict:
        uop = oper_dict['uop']
        gop = oper_dict['gop']
        bop = oper_dict['bop']
        bop = (bop - 1) % 6
        lft_chd_ans = conver_oper_dict_to_readable_dict(oper_dict['lft_chd'])
        rgt_chd_ans = conver_oper_dict_to_readable_dict(oper_dict['rgt_chd'])
        ans = convert_binary_oper_to_str(bop, lft_chd_ans, rgt_chd_ans)
        ans = convert_group_oper_to_str(gop, ans)
        ans = convert_unary_oper_to_str(uop, ans)
    else:
        raise ValueError(f"Unrecognized oper_dict: {oper_dict}")
    return ans


if __name__ == '__main__':
    baseline_loss_dict = [ # testing on 15.p1
                {'gop': 3, 'uop': 5, 'bop': 4, 'lft_chd': {'uop': 0, 'oper': 1}, 'rgt_chd': {'uop': 4, 'oper': 0}}, {'gop': 2, 'uop': 4, 'bop': 2, 'lft_chd': {'uop': 5, 'oper': 0}, 'rgt_chd': {'uop': 4, 'oper': 1}}]

    baseline_loss_dict = [{'gop': 1, 'uop': 1, 'chd': {'uop': 1, 'oper': 1}}, {'gop': 1, 'uop': 1, 'chd': {'uop': 1, 'oper': 0}}]

    # baseline_loss_dict = [ # testing on 15.p1
    #             {'gop': 1, 'uop': 5, 'bop': 3, 'lft_chd': {'uop': 4, 'oper': 0}, 'rgt_chd': {'uop': 5, 'oper': 0}}, {'gop': 1, 'uop': 4, 'bop': 2, 'lft_chd': {'uop': 2, 'oper': 3}, 'rgt_chd': {'uop': 1, 'oper': 6}}]

    # baseline_loss_dict = [{'gop': 3, 'uop': 0, 'bop': 16, 'lft_chd': {'uop': 2, 'oper': 1}, 'rgt_chd': {'gop': 2, 'uop': 2, 'bop': 2, 'lft_chd': {'uop': 0, 'oper': 1}, 'rgt_chd': {'uop': 4, 'oper': 5}}}, {'gop': 2, 'uop': 4, 'bop': 2, 'lft_chd': {'uop': 5, 'oper': 5}, 'rgt_chd': {'uop': 5, 'oper': 1}}]

    # baseline_loss_dict = [{'gop': 2, 'uop': 0, 'bop': 3, 'lft_chd': {'uop': 3, 'oper': 1}, 'rgt_chd': {'uop': 5, 'oper': 1}}, {'gop': 2, 'uop': 0, 'bop': 3, 'lft_chd': {'uop': 3, 'oper': 1}, 'rgt_chd': {'uop': 5, 'oper': 0}}]
    #
    baseline_loss_dict = [ # testing on 15.p2 --- model_4
                {'gop': 2, 'uop': 3, 'bop': 10, 'lft_chd': {'gop': 1, 'uop': 1, 'bop': 1, 'lft_chd': {'uop': 0, 'oper': 6}, 'rgt_chd': {'uop': 5, 'oper': 4}}, 'rgt_chd': {'uop': 5, 'oper': 4}}, {'gop': 1, 'uop': 0, 'bop': 4, 'lft_chd': {'uop': 1, 'oper': 2}, 'rgt_chd': {'uop': 0, 'oper': 5}}, {'gop': 2, 'uop': 0, 'bop': 3, 'lft_chd': {'uop': 3, 'oper': 1}, 'rgt_chd': {'uop': 5, 'oper': 1}}, {'gop': 2, 'uop': 0, 'bop': 3, 'lft_chd': {'uop': 3, 'oper': 1}, 'rgt_chd': {'uop': 5, 'oper': 0}}]

    baseline_loss_dict = [{'gop': 1, 'uop': 1, 'bop': 5, 'lft_chd': {'uop': 0, 'oper': 1}, 'rgt_chd': {'uop': 3, 'oper': 1}}, {'gop': 1, 'uop': 1, 'bop': 5, 'lft_chd': {'uop': 0, 'oper': 0}, 'rgt_chd': {'uop': 1, 'oper': 1}}]

    baseline_loss_dict =[{'gop': 1, 'uop': 4, 'bop': 15, 'lft_chd': {'uop': 2, 'oper': 0}, 'rgt_chd': {'gop': 1, 'uop': 1, 'chd': {'uop': 3, 'oper': 1}}}, {'gop': 2, 'uop': 2, 'bop': 3, 'lft_chd': {'uop': 4, 'oper': 0}, 'rgt_chd': {'uop': 1, 'oper': 0}}]

    baseline_loss_dict = [{'gop': 2, 'uop': 2, 'bop': 11, 'lft_chd': {'gop': 2, 'uop': 4, 'bop': 5, 'lft_chd': {'uop': 3, 'oper': 5}, 'rgt_chd': {'uop': 3, 'oper': 4}}, 'rgt_chd': {'uop': 3, 'oper': 4}}]

    baseline_loss_dict = [{'gop': 0, 'uop': 4, 'bop': 1, 'lft_chd': {'uop': 5, 'oper': 6}, 'rgt_chd': {'uop': 5, 'oper': 6}}]

    baseline_loss_dict = [ # Val acc = 0.8667025566101074, Test acc = 704 # 300 epochs
                {'gop': 0, 'uop': 4, 'bop': 1, 'lft_chd': {'uop': 5, 'oper': 6}, 'rgt_chd': {'uop': 5, 'oper': 6}},
                     {'gop': 0, 'uop': 4, 'bop': 3, 'lft_chd': {'uop': 4, 'oper': 2}, 'rgt_chd': {'uop': 2, 'oper': 1}}, ]

    baseline_loss_dict = [{
    'gop': 0, 'uop': 4, 'chd': {'uop': 0, 'oper': 2}},
    {'gop': 0, 'uop': 4, 'chd': {'uop': 0, 'oper': 6}},
    {'gop': 0, 'uop': 4, 'chd': {'uop': 1, 'oper': 6}},
    {'gop': 0, 'uop': 4, 'chd': {'uop': 0, 'oper': 6}},
    {'gop': 1, 'uop': 4, 'bop': 9, 'lft_chd': {'gop': 0, 'uop': 1, 'chd': {'uop': 0, 'oper': 3}}, 'rgt_chd': {'uop': 1, 'oper': 4}},
    {'gop': 0, 'uop': 4, 'chd': {'uop': 1, 'oper': 3}},
    {'gop': 0, 'uop': 4, 'chd': {'uop': 1, 'oper': 3}},
    {'gop': 0, 'uop': 4, 'chd': {'uop': 1, 'oper': 6}},
    {'gop': 1, 'uop': 4, 'bop': 7, 'lft_chd': {'gop': 0, 'uop': 5, 'bop': 3, 'lft_chd': {'uop': 5, 'oper': 2}, 'rgt_chd': {'uop': 5, 'oper': 2}}, 'rgt_chd': {'uop': 0, 'oper': 6}},
    {'gop': 1, 'uop': 4, 'bop': 7, 'lft_chd': {'gop': 0, 'uop': 0, 'bop': 4, 'lft_chd': {'uop': 3, 'oper': 5}, 'rgt_chd': {'uop': 1, 'oper': 5}}, 'rgt_chd': {'uop': 1, 'oper': 6}}
]

    baseline_loss_dict = [ # 0.8411913514137268; 0.739; test_v2_tree_2_losses_test_expand_first_level_3_layer_pnpp_m1
                                     {'gop': 0, 'uop': 4, 'chd': {'uop': 0, 'oper': 6}}] + [{'gop': 0, 'uop': 4, 'chd': {'uop': 1, 'oper': 3}},]

    baseline_loss_dict = [ # 11.p3 test_v2_tree_2_losses_model_test_model_6; 0.733 v.s. 0.945
                {'gop': 1, 'uop': 4, 'chd': {'uop': 2, 'oper': 1}}, {'gop': 0, 'uop': 1, 'bop': 2, 'lft_chd': {'uop': 1, 'oper': 0}, 'rgt_chd': {'uop': 1, 'oper': 1}}]

    baseline_loss_dict = [{'gop': 2, 'uop': 1, 'bop': 1, 'lft_chd': {'uop': 3, 'oper': 0}, 'rgt_chd': {'uop': 1, 'oper': 2}}]

    baseline_loss_dict = [{'gop': 2, 'uop': 1, 'bop': 1, 'lft_chd': {'uop': 3, 'oper': 0}, 'rgt_chd': {'uop': 1, 'oper': 2}}, {'gop': 2, 'uop': 1, 'bop': 1, 'lft_chd': {'uop': 2, 'oper': 1}, 'rgt_chd': {'uop': 1, 'oper': 1}}]

    baseline_loss_dict = [{'gop': 0, 'uop': 4, 'chd': {'uop': 1, 'oper': 6}},] + [{'gop': 1, 'uop': 4, 'bop': 7, 'lft_chd': {'gop': 0, 'uop': 0, 'bop': 4, 'lft_chd': {'uop': 3, 'oper': 5}, 'rgt_chd': {'uop': 1, 'oper': 5}}, 'rgt_chd': {'uop': 1, 'oper': 6}}]

    baseline_loss_dict = [{'gop': 2, 'uop': 4, 'bop': 4, 'lft_chd': {'uop': 3, 'oper': 5}, 'rgt_chd': {'uop': 3, 'oper': 5}},]

    baseline_loss_dict = [ {'gop': 1, 'uop': 1, 'chd': {'uop': 4, 'oper': 2}}]

    baseline_loss_dict = [{'gop': 1, 'uop': 3, 'chd': {'uop': 4, 'oper': 4}}]

    baseline_loss_dict = [{'gop': 1, 'uop': 2, 'bop': 4, 'lft_chd': {'uop': 2, 'oper': 4}, 'rgt_chd': {'uop': 1, 'oper': 6}}]

    baseline_loss_dict = [{'gop': 0, 'uop': 3, 'bop': 3, 'lft_chd': {'uop': 4, 'oper': 4}, 'rgt_chd': {'uop': 4, 'oper': 1}}, ]
    baseline_loss_dict = [
                {'gop': 0, 'uop': 3, 'bop': 3, 'lft_chd': {'uop': 4, 'oper': 4}, 'rgt_chd': {'uop': 4, 'oper': 1}},
                {'gop': 2, 'uop': 1, 'bop': 1, 'lft_chd': {'uop': 2, 'oper': 1},
                 'rgt_chd': {'uop': 1, 'oper': 1}},
            {'gop': 2, 'uop': 1, 'bop': 1, 'lft_chd': {'uop': 3, 'oper': 0}, 'rgt_chd': {'uop': 1, 'oper': 2}}]

    baseline_loss_dict = [{'gop': 3, 'uop': 3, 'chd': {'uop': 0, 'oper': 3}}]
    baseline_loss_dict = [{'gop': 2, 'uop': 4, 'bop': 3, 'lft_chd': {'uop': 0, 'oper': 4}, 'rgt_chd': {'uop': 4, 'oper': 6}}, {'gop': 1, 'uop': 3, 'chd': {'uop': 4, 'oper': 4}}, {'gop': 2, 'uop': 4, 'bop': 3, 'lft_chd': {'uop': 4, 'oper': 0}, 'rgt_chd': {'uop': 3, 'oper': 0}}]

    baseline_loss_dict = [{'gop': 1, 'uop': 1, 'chd': {'uop': 5, 'oper': 0}},
                                  {'gop': 1, 'uop': 1, 'chd': {'uop': 1, 'oper': 2}}]

    baseline_loss_dict = [{'gop': 2, 'uop': 4, 'chd': {'uop': 1, 'oper': 1}},
                                  {'gop': 2, 'uop': 4, 'chd': {'uop': 2, 'oper': 1}}]

    baseline_loss_dict = [{'gop': 2, 'uop': 4, 'bop': 3, 'lft_chd': {'uop': 2, 'oper': 5}, 'rgt_chd': {'uop': 0, 'oper': 3}}]

    baseline_loss_dict = [{'gop': 1, 'uop': 4, 'bop': 5, 'lft_chd': {'uop': 2, 'oper': 1}, 'rgt_chd': {'uop': 2,
                                                                                                               'oper': 0}},
                                  {'gop': 1, 'uop': 0, 'bop': 2, 'lft_chd': {'uop': 2, 'oper': 1}, 'rgt_chd': {'uop': 0,
                                                                                                               'oper': 0}},
                                  {'gop': 1, 'uop': 4, 'bop': 5, 'lft_chd': {'uop': 3, 'oper': 1}, 'rgt_chd': {'uop': 5, 'oper': 1}}]
    baseline_loss_dict = [{'gop': 1, 'uop': 1, 'chd': {'uop': 2, 'oper': 1}}, {'gop': 1, 'uop': 1, 'chd': {'uop': 0, 'oper': 2}}]

    baseline_loss_dict = [{'gop': 0, 'uop': 1, 'chd': {'uop': 1, 'oper': 1}}, {'gop': 1, 'uop': 1, 'bop': 5, 'lft_chd': {'uop': 1, 'oper': 4}, 'rgt_chd': {'uop': 2, 'oper': 2}}]
    baseline_loss_dict = [{'gop': 2, 'uop': 4, 'chd': {'uop': 3, 'oper': 1}}, {'gop': 2, 'uop': 4, 'chd': {'uop': 3, 'oper': 0}}, {'gop': 2, 'uop': 4, 'chd': {'uop': 3, 'oper': 4}}]
    baseline_loss_dict = [{'gop': 1, 'uop': 3, 'chd': {'uop': 4, 'oper': 1}}, {'gop': 1, 'uop': 4, 'chd': {'uop': 2, 'oper': 1}}]

    baseline_loss_dict = [{'gop': 2, 'uop': 4, 'chd': {'uop': 0,
                                                               'oper': 5}}, {'gop': 1, 'uop': 1, 'chd': {'uop': 4,
                                                                                                         'oper': 1}}, {'gop': 1, 'uop': 1, 'chd': {'uop': 2, 'oper': 2}}]

    baseline_loss_dict =  [{'gop': 0, 'uop': 4, 'chd': {'uop': 3, 'oper': 1}}]
    baseline_loss_dict = [{'gop': 3, 'uop': 3, 'chd': {'uop': 0, 'oper': 3}}]
    # (mean(cartesian(N, (-N)))) ^ 2

    baseline_loss_dict =  [{'gop': 1, 'uop': 4, 'bop': 7, 'lft_chd': {'gop': 0, 'uop': 0, 'bop': 4, 'lft_chd': {'uop': 3, 'oper': 5}, 'rgt_chd': {'uop': 1, 'oper': 5}}, 'rgt_chd': {'uop': 1, 'oper': 6}}] + [{'gop': 0, 'uop': 4, 'chd': {'uop': 1, 'oper': 6}},]

    baseline_loss_dict = [ # 11.p3 test_v2_tree_2_losses_model_test_model_6; 0.733 v.s. 0.945
                {'gop': 1, 'uop': 4, 'chd': {'uop': 2, 'oper': 1}}, {'gop': 0, 'uop': 1, 'bop': 2, 'lft_chd': {'uop': 1, 'oper': 0}, 'rgt_chd': {'uop': 1, 'oper': 1}}]

    baseline_loss_dict =  [ # 0.8411913514137268; 0.739; test_v2_tree_2_losses_test_expand_first_level_3_layer_pnpp_m1
                                     {'gop': 0, 'uop': 4, 'chd': {'uop': 0, 'oper': 6}}] + [{'gop': 0, 'uop': 4, 'chd': {'uop': 1, 'oper': 3}},]

    baseline_loss_dict = [{'gop': 2, 'uop': 1, 'bop': 1, 'lft_chd': {'uop': 3, 'oper': 0}, 'rgt_chd': {'uop': 1, 'oper': 2}}, {'gop': 2, 'uop': 1, 'bop': 1, 'lft_chd': {'uop': 2, 'oper': 1}, 'rgt_chd': {'uop': 1, 'oper': 1}}]


    # (mean(cartesian(P, (N) ^ 2))) ^ 2
    # gop + uop; orthogonal; or just merge them into a joint distribution


# (2*sum((P)^2+P)) (2*mean(cartesian(orth(P),centralize(N))))


    # all_dicts = [{'gop': 3, 'uop': 2, 'bop': 9, 'lft_chd': {'gop': 1, 'uop': 2, 'bop': 3, 'lft_chd': {'uop': 3, 'oper': 1}, 'rgt_chd': {'uop': 5, 'oper': 1}}, 'rgt_chd': {'uop': 2, 'oper': 5}}, {'gop': 3, 'uop': 1, 'bop': 13, 'lft_chd': {'uop': 5, 'oper': 6}, 'rgt_chd': {'gop': 2, 'uop': 0, 'bop': 3, 'lft_chd': {'uop': 5, 'oper': 4}, 'rgt_chd': {'uop': 3, 'oper': 6}}}, {'gop': 2, 'uop': 4, 'bop': 2, 'lft_chd': {'uop': 4, 'oper': 6}, 'rgt_chd': {'uop': 4, 'oper': 2}}, {'gop': 2, 'uop': 2, 'bop': 17, 'lft_chd': {'uop': 5, 'oper': 1}, 'rgt_chd': {'gop': 0, 'uop': 3, 'bop': 1, 'lft_chd': {'uop': 2, 'oper': 2}, 'rgt_chd': {'uop': 3, 'oper': 0}}}, {'gop': 2, 'uop': 0, 'bop': 1, 'lft_chd': {'uop': 4, 'oper': 2}, 'rgt_chd': {'uop': 2, 'oper': 1}}, {'gop': 0, 'uop': 2, 'bop': 4, 'lft_chd': {'uop': 4, 'oper': 0}, 'rgt_chd': {'uop': 1, 'oper': 1}}, {'gop': 3, 'uop': 0, 'chd': {'uop': 4, 'oper': 1}}, {'gop': 1, 'uop': 1, 'bop': 9, 'lft_chd': {'gop': 0, 'uop': 1, 'chd': {'uop': 2, 'oper': 5}}, 'rgt_chd': {'uop': 1, 'oper': 3}}, {'gop': 0, 'uop': 4, 'bop': 9, 'lft_chd': {'gop': 1, 'uop': 1, 'bop': 2, 'lft_chd': {'uop': 2, 'oper': 1}, 'rgt_chd': {'uop': 1, 'oper': 0}}, 'rgt_chd': {'uop': 3, 'oper': 3}}, {'gop': 2, 'uop': 0, 'bop': 5, 'lft_chd': {'uop': 5, 'oper': 0}, 'rgt_chd': {'uop': 2, 'oper': 1}}]

    # [{'gop': 2, 'uop': 3, 'bop': 9,
    #   'lft_chd': {'gop': 0, 'uop': 4, 'bop': 3, 'lft_chd': {'uop': 5, 'oper': 1}, 'rgt_chd': {'uop': 4, 'oper': 4}},
    #   'rgt_chd': {'uop': 2, 'oper': 4}}, {'gop': 2, 'uop': 3, 'bop': 9,
    #                                       'lft_chd': {'gop': 0, 'uop': 4, 'bop': 3, 'lft_chd': {'uop': 5, 'oper': 1},
    #                                                   'rgt_chd': {'uop': 4, 'oper': 4}},
    #                                       'rgt_chd': {'uop': 2, 'oper': 4}}]
    # 0.5373119115829468
    # [{'gop': 1, 'uop': 5, 'bop': 7,
    #   'lft_chd': {'gop': 0, 'uop': 0, 'bop': 4, 'lft_chd': {'uop': 4, 'oper': 5}, 'rgt_chd': {'uop': 0, 'oper': 5}},
    #   'rgt_chd': {'uop': 0, 'oper': 1}}, {'gop': 1, 'uop': 5, 'bop': 7,
    #                                       'lft_chd': {'gop': 0, 'uop': 0, 'bop': 4, 'lft_chd': {'uop': 4, 'oper': 5},
    #                                                   'rgt_chd': {'uop': 0, 'oper': 5}},
    #                                       'rgt_chd': {'uop': 0, 'oper': 1}}]
    # 0.5371932983398438
    # [{'gop': 1, 'uop': 5, 'bop': 7,
    #   'lft_chd': {'gop': 0, 'uop': 0, 'bop': 4, 'lft_chd': {'uop': 4, 'oper': 5}, 'rgt_chd': {'uop': 0, 'oper': 5}},
    #   'rgt_chd': {'uop': 0, 'oper': 1}}, {'gop': 1, 'uop': 3, 'chd': {'uop': 2, 'oper': 6}}]
    # 0.5335408449172974
    # [{'gop': 1, 'uop': 5, 'bop': 7,
    #   'lft_chd': {'gop': 0, 'uop': 0, 'bop': 4, 'lft_chd': {'uop': 4, 'oper': 5}, 'rgt_chd': {'uop': 0, 'oper': 5}},
    #   'rgt_chd': {'uop': 0, 'oper': 1}},
    #  {'gop': 1, 'uop': 1, 'bop': 10, 'lft_chd': {'gop': 2, 'uop': 0, 'chd': {'uop': 2, 'oper': 6}},
    #   'rgt_chd': {'uop': 1, 'oper': 4}}]
    # 0.5315394401550293
    # [{'gop': 1, 'uop': 5, 'bop': 7,
    #   'lft_chd': {'gop': 0, 'uop': 0, 'bop': 4, 'lft_chd': {'uop': 4, 'oper': 5}, 'rgt_chd': {'uop': 0, 'oper': 5}},
    #   'rgt_chd': {'uop': 0, 'oper': 1}}, {'gop': 2, 'uop': 3, 'bop': 9,
    #                                       'lft_chd': {'gop': 0, 'uop': 4, 'bop': 3, 'lft_chd': {'uop': 5, 'oper': 1},
    #                                                   'rgt_chd': {'uop': 4, 'oper': 4}},
    #                                       'rgt_chd': {'uop': 2, 'oper': 4}}]
    # 0.5097192525863647
    # [{'gop': 2, 'uop': 3, 'bop': 9,
    #   'lft_chd': {'gop': 0, 'uop': 4, 'bop': 3, 'lft_chd': {'uop': 5, 'oper': 1}, 'rgt_chd': {'uop': 4, 'oper': 4}},
    #   'rgt_chd': {'uop': 2, 'oper': 4}},
    #  {'gop': 1, 'uop': 1, 'bop': 10, 'lft_chd': {'gop': 2, 'uop': 0, 'chd': {'uop': 2, 'oper': 6}},
    #   'rgt_chd': {'uop': 1, 'oper': 4}}]
    # 0.5046136379241943
    # [{'gop': 1, 'uop': 1, 'bop': 10, 'lft_chd': {'gop': 2, 'uop': 0, 'chd': {'uop': 2, 'oper': 6}},
    #   'rgt_chd': {'uop': 1, 'oper': 4}}, {'gop': 1, 'uop': 3, 'chd': {'uop': 2, 'oper': 6}}]
    # 0.4973076581954956
    # [{'gop': 2, 'uop': 3, 'bop': 9,
    #   'lft_chd': {'gop': 0, 'uop': 4, 'bop': 3, 'lft_chd': {'uop': 5, 'oper': 1}, 'rgt_chd': {'uop': 4, 'oper': 4}},
    #   'rgt_chd': {'uop': 2, 'oper': 4}}, {'gop': 1, 'uop': 3, 'chd': {'uop': 2, 'oper': 6}}]
    # 0.48357856273651123
    # [{'gop': 1, 'uop': 1, 'bop': 10, 'lft_chd': {'gop': 2, 'uop': 0, 'chd': {'uop': 2, 'oper': 6}},
    #   'rgt_chd': {'uop': 1, 'oper': 4}},
    #  {'gop': 1, 'uop': 1, 'bop': 10, 'lft_chd': {'gop': 2, 'uop': 0, 'chd': {'uop': 2, 'oper': 6}},
    #   'rgt_chd': {'uop': 1, 'oper': 4}}]
    # 0.4736548066139221
    # [{'gop': 1, 'uop': 3, 'chd': {'uop': 2, 'oper': 6}}, {'gop': 1, 'uop': 3, 'chd': {'uop': 2, 'oper': 6}}]

    #
    # for dict in all_dicts:
    #     print(dict)

    # mean(N - -max(cross(inv(P), P ^ 2))) ^ 2
    for oper_dict in baseline_loss_dict:
        ans = conver_oper_dict_to_readable_dict(oper_dict)
        print(ans)