"""
Ascend NPU / host memory profiling helpers for training scripts.

Reads HBM via `npu-smi` (primary) or ACL (optional). Use log_memory(stage) at key points.
"""
import os
import re
import subprocess
import time

import psutil

_PROCESS = psutil.Process(os.getpid())
_BASELINE_NPU_MB = None
_BASELINE_RSS_MB = None
_ENABLED = os.environ.get('MEM_PROFILE', '1').lower() in ('1', 'true', 'yes')
_LOG_ALL_RANKS = os.environ.get('MEM_PROFILE_ALL_RANKS', '0').lower() in ('1', 'true', 'yes')


def _rank():
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))


def _should_log():
    if not _ENABLED:
        return False
    if _LOG_ALL_RANKS:
        return True
    return _rank() == 0


def _device_id():
    return os.environ.get('ASCEND_DEVICE_ID', os.environ.get('WORKER_DEVICE_ID', '0'))


def _run_cmd(cmd, timeout=15):
    try:
        proc = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or '', proc.stderr or ''
    except Exception as exc:
        return -1, '', str(exc)


def _parse_hbm_from_info(text):
    """Parse `HBM-Usage(MB) : used / total` from npu-smi info table."""
    patterns = [
        r'HBM[- ]Usage\(MB\)\s*:\s*(\d+)\s*/\s*(\d+)',
        r'HBM\s+Usage\(MB\)\s*:\s*(\d+)\s*/\s*(\d+)',
        r'HBM\s+Used\s*:\s*(\d+)\s*MiB',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            if len(m.groups()) == 2:
                return int(m.group(1)), int(m.group(2))
            return int(m.group(1)), None
    return None, None


def _parse_hbm_from_usages(text, capacity_mb=None):
    """Parse HBM usage rate from `npu-smi info -t usages`."""
    m = re.search(r'HBM Usage Rate\(%\)\s*:\s*(\d+)', text, re.IGNORECASE)
    if m and capacity_mb:
        rate = int(m.group(1)) / 100.0
        used = int(capacity_mb * rate)
        return used, capacity_mb
    return None, None


def _parse_hbm_capacity(text):
    m = re.search(r'HBM Capacity\(MB\)\s*:\s*(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def get_npu_hbm_mb(device_id=None):
    """
    Returns (used_mb, total_mb, source_str). Values may be None if query fails.
    """
    if device_id is None:
        device_id = _device_id()

    # 1) npu-smi info (HBM used/total on chip row)
    rc, out, err = _run_cmd(['npu-smi', 'info', '-i', str(device_id)])
    if rc == 0 and out:
        used, total = _parse_hbm_from_info(out)
        if used is not None:
            return used, total, 'npu-smi info'

    # 2) memory capacity + usages rate
    rc, mem_out, _ = _run_cmd(['npu-smi', 'info', '-t', 'memory', '-i', str(device_id), '-c', '0'])
    capacity = _parse_hbm_capacity(mem_out) if rc == 0 else None
    rc2, usage_out, _ = _run_cmd(['npu-smi', 'info', '-t', 'usages', '-i', str(device_id), '-c', '0'])
    if rc2 == 0 and usage_out:
        used, total = _parse_hbm_from_usages(usage_out, capacity)
        if used is not None:
            return used, total, 'npu-smi usages'

    # 3) ACL runtime (optional)
    try:
        import acl
        ret = acl.rt.set_device(int(device_id))
        if ret == 0:
            _, mem_info = acl.rt.get_mem_info(1)
            used_mb = mem_info.used // (1024 * 1024)
            total_mb = mem_info.total // (1024 * 1024)
            return used_mb, total_mb, 'acl.rt.get_mem_info'
    except Exception:
        pass

    return None, None, 'unavailable(%s)' % (err[:80] if err else 'no npu-smi')


def get_host_rss_mb():
    return _PROCESS.memory_info().rss / (1024 * 1024)


def estimate_tf_variables_mb():
    """Estimate float32 size of all global variables (graph built, may not be allocated yet)."""
    import tensorflow as tf
    if tf.__version__.startswith('2'):
        import tensorflow.compat.v1 as tf

    emb_mb = 0.0
    dense_mb = 0.0
    slot_mb = 0.0
    other_mb = 0.0
    for var in tf.global_variables():
        shape = var.get_shape()
        if not shape.is_fully_defined():
            continue
        nbytes = shape.num_elements() * 4
        name = var.name.lower()
        mb = nbytes / (1024 * 1024)
        if any(k in name for k in ('_m_hash', '_v_hash', '/m_slot', '/v_slot', '/m/', '/v/')):
            slot_mb += mb
        elif any(k in name for k in ('adam', 'slot', 'lazyadam', 'sparse_adam')):
            slot_mb += mb
        elif 'merged_embedding_table' in name or 'embedding_table' in name:
            emb_mb += mb
        elif any(k in name for k in ('cross_', 'deep_', 'model/', 'dense')):
            dense_mb += mb
        else:
            other_mb += mb
    return {
        'embedding_mb': emb_mb,
        'optimizer_slot_mb': slot_mb,
        'dense_mb': dense_mb,
        'other_mb': other_mb,
        'total_mb': emb_mb + slot_mb + dense_mb + other_mb,
    }


def estimate_merged_tables_mb(merged_tables, embedding_dim):
    total_rows = sum(getattr(mt, 'global_capacity', 0) for mt in merged_tables)
    emb_mb = total_rows * embedding_dim * 4 / (1024 * 1024)
    return total_rows, emb_mb


def log_memory(stage, extra=None, use_baseline=True, merged_tables=None, embedding_dim=None):
    """
    Print one memory snapshot line. Call at each training lifecycle stage.

    extra: optional dict printed after main line.
    merged_tables: optional list for theoretical embedding MB estimate.
    """
    global _BASELINE_NPU_MB, _BASELINE_RSS_MB
    if not _should_log():
        return

    rank = _rank()
    dev = _device_id()
    ts = time.strftime('%H:%M:%S')
    npu_used, npu_total, npu_src = get_npu_hbm_mb(dev)
    rss_mb = get_host_rss_mb()

    if use_baseline and _BASELINE_NPU_MB is None and npu_used is not None:
        _BASELINE_NPU_MB = npu_used
        _BASELINE_RSS_MB = rss_mb

    delta_npu = None
    delta_rss = None
    if npu_used is not None and _BASELINE_NPU_MB is not None:
        delta_npu = npu_used - _BASELINE_NPU_MB
    if _BASELINE_RSS_MB is not None:
        delta_rss = rss_mb - _BASELINE_RSS_MB

    tf_est = estimate_tf_variables_mb()
    parts = [
        '[MEM_PROFILE]',
        'rank=%s' % rank,
        'dev=%s' % dev,
        'stage=%s' % stage,
        'time=%s' % ts,
    ]
    if npu_used is not None:
        line = 'NPU_HBM=%dMB' % npu_used
        if npu_total is not None:
            line += '/%dMB' % npu_total
        if delta_npu is not None:
            line += '(+%dMB)' % delta_npu
        line += '[%s]' % npu_src
        parts.append(line)
    else:
        parts.append('NPU_HBM=NA[%s]' % npu_src)

    rss_line = 'host_RSS=%dMB' % int(rss_mb)
    if delta_rss is not None:
        rss_line += '(+%dMB)' % int(delta_rss)
    parts.append(rss_line)

    parts.append('tf_var_est=%.1fMB(emb=%.1f slot=%.1f dense=%.1f)' % (
        tf_est['total_mb'], tf_est['embedding_mb'],
        tf_est['optimizer_slot_mb'], tf_est['dense_mb']))

    if merged_tables is not None and embedding_dim is not None:
        rows, emb_mb = estimate_merged_tables_mb(merged_tables, embedding_dim)
        parts.append('merged_table_est=%.1fMB(rows=%d)' % (emb_mb, rows))

    print(' | '.join(parts), flush=True)
    if extra:
        print('  [MEM_EXTRA] %s' % extra, flush=True)


def log_memory_banner(title):
    if _should_log():
        print('\n' + '=' * 72, flush=True)
        print('[MEM_PROFILE] %s' % title, flush=True)
        print('=' * 72, flush=True)
