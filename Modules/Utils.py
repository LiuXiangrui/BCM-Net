import struct

GOP_SIZE = 8


def get_ref_idx(slice_idx: int, num_slices: int) -> tuple:
    """
    Get forward and backward reference indices.
    :param slice_idx: slice index of current frame within the slices.
    :param num_slices: number of slices.
    :return forward_ref_idx: forward reference index.
    :return backward_ref_idx: backward reference index.
    """
    assert slice_idx < num_slices, "frame index {} >= number of frames {}".format(slice_idx, num_slices)
    if slice_idx == 0:
        return -1, -1
    if slice_idx % GOP_SIZE == 0:  # POC = 8, 16, ...
        return slice_idx - GOP_SIZE, -1
    ref_map = {
        1: [0, 2],
        2: [0, 4],
        3: [2, 4],
        4: [0, 8],
        5: [4, 6],
        6: [4, 8],
        7: [6, 8],
    }
    num_gop = slice_idx // GOP_SIZE
    idx_within_gop = slice_idx % GOP_SIZE
    forward_ref_idx, backward_ref_idx = ref_map[idx_within_gop]
    if forward_ref_idx != -1:
        forward_ref_idx += num_gop * GOP_SIZE
    if backward_ref_idx != -1:
        backward_ref_idx += num_gop * GOP_SIZE
    if backward_ref_idx >= num_slices:
        backward_ref_idx = -1
    return forward_ref_idx, backward_ref_idx


def calculate_decompression_order(num_slices: int) -> list:
    """
    Calculate the decompression order of slices.
    :param num_slices: number of slices.
    :return slices_indices: decompression order of slices.
    """
    assert num_slices > 0
    slices_indices = [0]

    base = [8, 4, 2, 6, 1, 3, 5, 7]

    num_gop = (num_slices - 1) // GOP_SIZE
    for n in range(num_gop):
        slices_indices += [i + n * GOP_SIZE for i in base]
    start_idx = num_gop * GOP_SIZE
    for i in base:
        if start_idx + i >= num_slices:
            continue
        slices_indices.append(start_idx + i)
    return slices_indices


def write_uintx(f, value: int, x: int) -> None:
    """
    Write unsigned integer with bit depth x to file.
    :param f: file handler.
    :param value: unsigned integer to be written.
    :param x: bit depth.
    """
    bit_depth_map = {
        8: 'B',
        16: 'H',
        32: 'I'
    }
    f.write(struct.pack(">{}".format(bit_depth_map[x]), value))


def write_bytes(f, values, fmt=">{:d}s") -> None:
    """
    Write bytes to file.
    :param f: file handler.
    :param values: bytes to be written.
    :param fmt: format of bytes.
    """
    if len(values) == 0:
        return
    f.write(struct.pack(fmt.format(len(values)), values))


def read_uintx(f, x: int) -> int:
    """
    Read unsigned integer with bit depth x from file.
    :param f: file handler.
    :param x: bit-depth.
    """
    bit_depth_map = {
        8: 'B',
        16: 'H',
        32: 'I'
    }
    return struct.unpack(">{}".format(bit_depth_map[x]), f.read(struct.calcsize(bit_depth_map[x])))[0]


def read_bytes(f, n, fmt=">{:d}s"):
    """
    Read bytes from file.
    :param f: file handler.
    :param n: number of bytes to be read.
    :param fmt: format of bytes.
    """
    return struct.unpack(fmt.format(n), f.read(n * struct.calcsize("s")))[0]
