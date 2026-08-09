"""Convert an X11 window dump to PNG.

CI captures the viewer with `xwd` so the window is uploaded as an artifact
rather than described by a statistic. A panel that stops drawing, renders
black, or lands in the wrong place looks identical from out here to one that
works -- the only cure is a picture somebody can open.
"""

import struct
import sys

from PIL import Image

data = open(sys.argv[1], "rb").read()
# XWD headers are big-endian u32s regardless of the host.
header = struct.unpack(">25I", data[:100])
header_size, width, height, bytes_per_line = header[0], header[4], header[5], header[12]
# The colormap follows the header, twelve bytes an entry.
pixels = data[header_size + header[19] * 12 :]
image = Image.frombytes(
    "RGBX", (width, height), pixels[: bytes_per_line * height], "raw", "BGRX", bytes_per_line
)
image.convert("RGB").save(sys.argv[2])
print(f"{sys.argv[2]}: {width}x{height}")
