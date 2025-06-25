import torch
import copy
import numpy as np
import os
import yaml
import cmath

def L2forTest(outputs,targets,obs_length):
    '''
    Evaluation.
    information: [N, 3]
    '''
    seq_length = outputs.shape[0]
    error = torch.norm(outputs-targets,p=2,dim=2)
    error_pred_length = error[obs_length-1:]
    error = torch.sum(error_pred_length)
    error_cnt = error_pred_length.numel()
    if error == 0:
        return 0,0,0,0,0,0
    final_error = torch.sum(error_pred_length[-1])
    final_error_cnt = error_pred_length[-1].numel()
    first_erro = torch.sum(error_pred_length[0])
    first_erro_cnt = error_pred_length[0].numel()
    return error.item(),error_cnt,final_error.item(),final_error_cnt,first_erro.item(),first_erro_cnt

def displacement_error(pred, pred_gt, mode='raw'):
    assert pred_gt.shape[1:] == pred.shape[1:]
    loss = pred_gt - pred
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred, pred_gt, mode='raw'):
    assert pred_gt.shape[1:] == pred.shape[1:]
    loss = pred_gt - pred
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)

def display_performance(perf_dict):
    print("==> Current Performances (ADE & FDE):")
    for a, b in perf_dict.items():
        c = copy.deepcopy(b)
        if isinstance(c, list):
            c[0] = np.round(c[0], 4)
            c[1] = np.round(c[1], 4)
        print("   ", a, c)

def load_arg(p, parser):
    # save arg
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            # default_arg = yaml.load(f,Loader=yaml.FullLoader)
            default_arg = yaml.safe_load(f)

        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s=1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False

def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_filepath):
        os.makedirs(args.model_filepath)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def balance_mapping(large_number,small_number):# 计算整除值
    # large_number = len(large_list)
    # small_number = len(small_list)
    quotient = large_number // small_number
    # 计算余数值
    remainder = large_number % small_number

    # 初始化映射结果的列表
    mapping = []
    currentIndex=0
    # small_number
    for right in range(small_number):
        # 计算当前小数项应该匹配的大数项数量
        current_match_count = quotient + (1 if remainder>0 else 0)
        current_match_count += currentIndex
        # 对于每个匹配的大数项进行添加
        for left in range(currentIndex,current_match_count):
            # 添加映射
            mapping.append([left,right])
        currentIndex = current_match_count
        remainder-=1
    return mapping

def update_probabilities(probabilities):
    # 确保输入是一个张量
    probabilities = torch.tensor(probabilities, dtype=torch.float32)
    probabilities = probabilities.clone().detach()
    # 保留概率大于阈值的值，并将其他值设置为0
    max_val = max(probabilities)
    min_val = min(probabilities)
    # 去掉最大值和最小值
    trimmed_arr = [x for x in probabilities if x != max_val and x != min_val]
    # 计算剩余值的平均值
    if len(trimmed_arr) == 0:
        return probabilities
    Threshold = sum(trimmed_arr) / len(trimmed_arr)
    # Threshold = torch.max(probabilities)
    updated_probs = torch.where(probabilities < Threshold, torch.zeros_like(probabilities), probabilities)

    # 计算剩余概率的总和
    remaining_sum = torch.sum(updated_probs)

    # 如果剩余总和为0，则返回原始数组（避免除以0的情况）
    if torch.isclose(remaining_sum, torch.tensor(0.0)):
        return probabilities

    # 重新调整剩余概率值的总和为1
    updated_probs /= remaining_sum

    return updated_probs.to('cuda')

def modifyArgsfile(file_path, key, value):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 查找并删除原始的cluster_num行
    cluster_num_line = None
    for i, line in enumerate(lines):
        if line.startswith(key):
            cluster_num_line = i
            break

    if cluster_num_line is not None:
        del lines[cluster_num_line]

    # 插入新的cluster_num行
    new_line = f'{key}: {value}\n'
    lines.insert(cluster_num_line, new_line)

    # 将修改后的内容写回文件
    with open(file_path, 'w') as file:
        file.writelines(lines)

    print(f'{key} has been updated to {value}')


def get_speed_and_angle(data, t):
    data = data.permute(1,0,2)
    data = data.cpu().detach().numpy()
    data_size = len(data)
    speed = np.zeros((data_size, 1)).astype(np.float32)  # Format: (speed, angle)
    angle = np.zeros((data_size, 1)).astype(np.float32)
    for i in range(len(data)):
        d_x, d_y = data[i, t] - data[i, t-1]
        speed[i],angle[i] = cmath.polar(complex(d_x, d_y))
    return torch.from_numpy(speed).to('cuda'), torch.from_numpy(angle).to('cuda')