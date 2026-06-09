import logging
import os
import re

import numpy as np


def setup_logger(rank=0, log_dir='./logs'):
    """
    配置并返回一个 logger，日志仅输出到文件（不输出到控制台）。
    文件名为: {log_dir}/training_rank_{rank}.log
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_rank_{rank}.log')

    logger = logging.getLogger(f'train_rank_{rank}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def _to_numpy(data):
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        return [_to_numpy(x) for x in data]
    if hasattr(data, 'numpy'):
        return data.numpy()
    return np.asarray(data)


def _format_values(arr, max_show):
    flat = np.asarray(arr).reshape(-1)
    n = flat.size
    if n == 0:
        return '[]'
    show = min(n, max_show)
    vals = ', '.join(f'{v!r}' for v in flat[:show]) # {v!r} 表示将v转换为字符串，并使用repr()函数进行格式化,字符串加引号，整数则不变
    if n > show:
        vals += f', ... (+{n - show} more)'
    return f'[{vals}]'


def log_section(logger, title):
    logger.info('=' * 72)
    logger.info(title)
    logger.info('=' * 72)


def log_array(logger, tag, data, max_show=None, indent=''):
    """记录 numpy / 可转 numpy 数据的 shape 与取值（过长时截断）。"""
    if max_show is None:
        max_show = int(os.environ.get('LOG_MAX_ELEMENTS', '64'))
    arr = _to_numpy(data)
    if arr is None:
        logger.info(f'{indent}{tag}: None')
        return
    if isinstance(arr, list):
        logger.info(f'{indent}{tag}: list len={len(arr)}')
        for i, item in enumerate(arr):
            log_array(logger, f'{tag}[{i}]', item, max_show=max_show, indent=indent + '  ')
        return
    arr = np.asarray(arr)
    logger.info(
        f'{indent}{tag}: shape={tuple(arr.shape)}, dtype={arr.dtype}, '
        f'values={_format_values(arr, max_show)}'
    )


def log_feed_inputs(logger, step, local_indices, labels, detail_slots=None):
    """记录训练一步的 placeholder 喂入数据。"""
    if detail_slots is None:
        detail_slots = int(os.environ.get('LOG_DETAIL_SLOTS', '5'))
    log_section(logger, f'[Step {step}] 1. 输入数据 (feed_dict)')
    log_array(logger, 'labels', labels)
    logger.info(f'  slot 总数: {len(local_indices)}')
    for i, indices in enumerate(local_indices):
        if i < detail_slots:
            log_array(logger, f'input_slot_{i}', indices, indent='  ')
        elif i == detail_slots:
            logger.info(f'  ... 其余 {len(local_indices) - detail_slots} 个 slot 仅记录 shape')
        if i >= detail_slots:
            arr = np.asarray(indices)
            logger.info(f'  input_slot_{i}: shape={tuple(arr.shape)}, dtype={arr.dtype}')


def log_trace_results(logger, step, trace_names, trace_values, detail_slots=None):
    """记录 sess.run 拉取的图内中间张量。"""
    if detail_slots is None:
        detail_slots = int(os.environ.get('LOG_DETAIL_SLOTS', '5'))
    log_section(logger, f'[Step {step}] 2. 图内计算过程 (embedding / all2all / key_process)')
    for name, value in zip(trace_names, trace_values):
        if 'sub_table_output_' in name or 'sub_table_emb_' in name:
            m = re.search(r'\.slot_(\d+)\.', name)
            if m:
                slot_idx = int(m.group(1))
            else:
                try:
                    slot_idx = int(name.rsplit('_', 1)[-1])
                except ValueError:
                    slot_idx = detail_slots
            if slot_idx >= detail_slots:
                arr = _to_numpy(value)
                if arr is not None:
                    logger.info(
                        f'  {name}: shape={tuple(np.asarray(arr).shape)}, '
                        f'dtype={np.asarray(arr).dtype} (values omitted)'
                    )
                continue
        log_array(logger, name, value, indent='  ')


def log_model_outputs(logger, step, outputs):
    """记录模型输出：batch_embedding、logits、loss 等。"""
    log_section(logger, f'[Step {step}] 3. 模型输出 (DCN + loss)')
    for tag, data in outputs.items():
        log_array(logger, tag, data, indent='  ')


def log_config(logger, **kwargs):
    log_section(logger, '训练配置')
    for k, v in kwargs.items():
        logger.info(f'  {k} = {v!r}')
