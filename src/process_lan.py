import numpy as np
import os
import re
def load_lan(filename, trunc_level):
    """
    Reads a .lan file and returns a list of levels truncated (or extended)
    to trunc_level.
    
    Each level is represented as a list of (epsilon, critical_value) tuples.
    The file is assumed to have header lines (starting with '#' or '%') indicating
    the beginning of a new level. Critical points for a level appear on subsequent lines.
    This is the standard format of .lan files.
    """
    levels = []
    current_points = []
    current_level_count = 0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#') or line.startswith('%'):
                # Header encountered: finish the previous level if present.
                if current_points:
                    levels.append(current_points)
                    current_points = []
                    current_level_count += 1
                    if current_level_count == trunc_level:
                        break
                continue
            # Only parse data if we haven't reached the truncation limit.
            if current_level_count < trunc_level:
                tokens = line.split()
                if len(tokens) != 2:
                    raise ValueError("Expected exactly two tokens per critical point line, got: " + str(tokens))
                eps = float(tokens[0])
                val = float(tokens[1])
                current_points.append((eps, val))
    # Append any remaining points as the last level.
    if current_points and current_level_count < trunc_level:
        levels.append(current_points)
        current_level_count += 1

    # If there are fewer than trunc_level levels, extend with empty lists.
    while current_level_count < trunc_level:
        levels.append([])
        current_level_count += 1

    return levels

def eval_landscape(points, t):
    """
    Evaluate a landscape function defined by a list of (epsilon, value), i.e. (x, landscape_level(x)) tuples
    at a given epsilon value t via linear interpolation. The function is assumed 
    to be zero outside the range of the provided points.
    """
    if not points:
        return 0.0
    # t is outside the range -> return 0.
    if t < points[0][0] or t > points[-1][0]:
        return 0.0
    # Find the interval [x0, x1] in which t lies.
    for i in range(1, len(points)):
        x0, y0 = points[i-1]
        x1, y1 = points[i]
        if x0 <= t <= x1:
            if t == x0:
                return y0
            if t == x1:
                return y1
            # Linear interpolation.
            return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
    return 0.0

def lan2array_batch(directory, trunc_level, grid=None, file_list=None):
    """
    Processes all .lan files in the directory and returns a 3D NumPy array
    of time series data and an ordering dictionary.
    
    The output 3D array has shape (N, trunc_level, T) where:
      - N = number of landscapes (files)
      - trunc_level = number of levels per landscape (as specified)
      - T = number of unique epsilon values across all files (or grid steps if grid is specified)
    The epsilon axis is common for all landscapes.
    
    Args:
        directory: path to directory containing .lan files
        trunc_level: number of levels to truncate to
        grid: if provided, uses uniform linspace from 0 to max epsilon with this many steps
              if None, uses all unique epsilon values (default behavior)
        file_list: optional list of filenames to process (subset). If None, process all matching '1.lan'.
    
    Also returns:
      - ordering: a dictionary mapping each filename (without full path)
                   to its index in the 3D array.
      - global_epsilon: the sorted list of all unique epsilon values or the linspace grid.
    """
    all_epsilons = set()

    # First, gather all .lan filenames.
    if file_list is None:
        file_list = []
        for fname in os.listdir(directory):
            if fname.endswith('1.lan'):
                file_list.append(fname)
    else:
        # Keep only those existing in directory and ending with '1.lan'.
        file_list = [f for f in sorted(file_list) if f.endswith('1.lan') and os.path.exists(os.path.join(directory, f))]

    file_list.sort()  # Ensure a consistent ordering

    # First pass: Accumulate epsilon values from all files.
    for fname in file_list:
        fpath = os.path.join(directory, fname)
        levels_data = load_lan(fpath, trunc_level)
        for level in levels_data:
            for (eps, _) in level:
                all_epsilons.add(eps)
    
    if grid is None:
        # Default behavior: use all unique epsilon values
        global_epsilon = sorted(all_epsilons)
    else:
        # Use uniform linspace from 0 to max epsilon
        max_epsilon = max(all_epsilons)
        global_epsilon = np.linspace(0, max_epsilon, grid)
    
    T = len(global_epsilon)

    landscapes = []    # to store the time series matrices for each file
    ordering = {}      # maps filename to its index in the 3D array

    # Second pass: For each file, evaluate the landscape at all global epsilon values.
    for idx, fname in enumerate(file_list):
        fpath = os.path.join(directory, fname)
        levels_data = load_lan(fpath, trunc_level)
        ts_matrix = np.zeros((trunc_level, T))
        for lev in range(trunc_level):
            pts = levels_data[lev]  # list of (eps, value) for this level (may be empty)
            for j, t in enumerate(global_epsilon):
                ts_matrix[lev, j] = eval_landscape(pts, t)
        landscapes.append(ts_matrix)
        ordering[fname] = idx

    # Stack into a 3D array: shape (N, trunc_level, T)
    result_array = np.stack(landscapes, axis=0)
    return result_array, ordering, global_epsilon

def lan2array(path2lan, trunc_level):
    """
    Processes a .lan file and returns a NumPy array
    representation of the landscape up to trunc_level.

    The output array has shape (num_crit_points, trunc_level) where:
      - trunc_level = number of levels per landscape (as specified)
      - num_crit_points = number of critical points in the landscape
    """
    all_critical = set()

    # First pass: Accumulate critical points from all levels.
    levels_data = load_lan(path2lan, trunc_level)
    for level in levels_data:
        for (crit, _) in level:
            all_critical.add(crit)
    lan_critical = sorted(all_critical)
    T = len(lan_critical)

    # Second pass: For each file, evaluate the landscape at all global critical points.
    landscape_array = np.zeros((T, trunc_level))
    for lev in range(trunc_level):
        pts = levels_data[lev]  # list of (eps, value) for this level (may be empty)
        for j, t in enumerate(lan_critical):
            landscape_array[j, lev] = eval_landscape(pts, t)
    return landscape_array, lan_critical