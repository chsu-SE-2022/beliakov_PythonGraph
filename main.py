import struct
import sys
from typing import Any

import numpy as np
import typing
from mpl_toolkits.axisartist import Axes
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import ndarray, dtype


class Header:
    # Width
    width: int
    # Height
    height: int
    # Horizontal scale
    stepx: float
    # Vertical scale
    stepy: float
    # Start of data by width
    xstart: int
    # Start of data by height
    ystart: int
    # End of data by width
    xend: int
    # End of data by height
    yend: int
    # Points by width
    wpoints: int
    # Points by height
    hpoints: int
    # Base height level
    level: float
    # Header length
    header_length: int

    def __init__(self, parts, length):
        self.width = int(parts[0])
        self.height = int(parts[1])
        self.stepx = float(parts[2])
        self.stepy = float(parts[3])
        self.xstart = int(parts[4])
        self.ystart = int(parts[5])
        self.xend = int(parts[6])
        self.yend = int(parts[7])
        self.wpoints = int(parts[8])
        self.hpoints = int(parts[9])
        self.level = float(parts[10])
        self.header_length = length


class Data:
    header: Header
    data: list[int]

    def __init__(self, header: Header, data: list[int]):
        self.header = header
        self.data = data


def parse_int_or_float(substr: str) -> int | float:
    try:
        substr = substr.replace(',', '.')
        return float(substr)
    except Exception:
        return int(substr)


def parse_header(raw: str, length: str) -> Header:
    parts: list[bytes] = raw.strip().split('|')
    parts.pop()
    parts = list(map(parse_int_or_float, parts))

    return Header(parts, int(length))


def parse_data(buffer: typing.BinaryIO) -> Data:
    header_raw: bytes = buffer.readline()
    header_len_raw: bytes = buffer.readline()
    header_str: str = header_raw.decode('utf-8')
    header_len_str: str = header_len_raw.decode('utf-8')
    header_parsed: Header = parse_header(header_str, header_len_str)
    buffer.seek(int(header_len_str))
    data_size = header_parsed.width * header_parsed.height
    data_raw = buffer.read(4 * data_size)
    data = np.array(struct.unpack(f'<{data_size}f', data_raw))
    data = np.array(list(map(lambda x: x, data)))
    return Data(header_parsed, data)


def draw_graph(data: Data):
    arr = np.reshape(data.data, [data.header.height, data.header.width])
    arr = np.rot90(arr)
    extents =[data.header.ystart, 
           data.header.yend * data.header.stepy, 
           data.header.xstart, 
           data.header.xend * data.header.stepx]
    plt.gca().set_yticks(np.arange(data.header.ystart, data.header.yend * data.header.stepy, 200))    
    graph = plt.imshow(arr, cmap='hot', interpolation=None, 
                       extent=extents)
    norm = mpl.colors.Normalize(vmin=np.average(arr), vmax=np.average(arr) + 4)
    bar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=graph.cmap), ax=graph.axes, label="Verbosity coefficient")
    bar.set_ticks(np.linspace(np.average(arr), np.average(arr) + 4, num=9))
    print(data.header.__dict__)
    # Bottom: np.average(arr)
    plt.savefig('foo.png')


def main(path: str):
    with open(path, "rb") as file:
        data: Data = parse_data(file)
        draw_graph(data)


if __name__ == "__main__":
    main(sys.argv[1])
