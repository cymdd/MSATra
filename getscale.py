import torch

def get_ScaleSeq(data):
    scale_array = torch.tensor([1, 2, 4])
    results = {}
    for length in scale_array:
        # 计算需要多少个完整的块和剩余的块
        num_full_blocks = len(data) // length
        remaining = len(data) % length
        # 创建一个空列表来存储当前长度的划分
        current_result = []
        # 添加完整的块
        for i in range(num_full_blocks):
            current_result.append(data[i * length:(i + 1) * length])
        # 如果有剩余，添加剩余的元素
        if remaining > 0:
            # 从张量a的开始借元素
            borrowed_elements = data[-length:]
            # 将剩余的元素和借来的元素合并
            current_result.append(borrowed_elements)
        # 将当前长度的结果添加到总结果字典中
        results[length.item()] = current_result
    return results

if __name__ == '__main__':
    data = torch.arange(7 * 48).view(7, 48)
    get_ScaleSeq(data)