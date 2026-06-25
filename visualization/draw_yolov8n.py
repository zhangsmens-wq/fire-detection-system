import sys
import os

# 1. 路径定位
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, ".."))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # --- Backbone (YOLOv8n 参数) ---
    to_Conv("stem", 320, 16, offset="(0,0,0)", to="(0,0,0)", height=80, depth=80, width=2, caption="Stem"),
    to_ConvRes("p2", 160, 32, offset="(2,0,0)", to="(stem-east)", height=60, depth=60, width=4, caption="C2f-P2"),
    to_ConvRes("p3", 80, 64, offset="(2,0,0)", to="(p2-east)", height=45, depth=45, width=6, caption="C2f-P3"),
    to_ConvRes("p4", 40, 128, offset="(2,0,0)", to="(p3-east)", height=30, depth=30, width=8, caption="C2f-P4"),
    to_ConvRes("p5", 20, 256, offset="(2,0,0)", to="(p4-east)", height=20, depth=20, width=10, caption="C2f-P5"),
    to_Pool("sppf", offset="(1.5,0,0)", to="(p5-east)", height=20, depth=20, width=12, caption="SPPF"),

    # --- Head (重点：坐标拉开，杜绝重叠) ---
    to_Conv("h_s", 80, 64, offset="(4, 4, 0)", to="(sppf-east)", height=45, depth=45, width=4, caption="Detect(S)"),
    to_Conv("h_m", 40, 128, offset="(6, 0, 0)", to="(sppf-east)", height=30, depth=30, width=4, caption="Detect(M)"),
    to_Conv("h_l", 20, 256, offset="(4, -4, 0)", to="(sppf-east)", height=20, depth=20, width=4, caption="Detect(L)"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    tex_file = namefile + '.tex'

    # 第一步：生成 .tex 文件
    with open(tex_file, 'w') as f:
        for x in arch:
            if isinstance(x, str):
                f.write(x)
            else:
                f.write(x.get_custom_block() if hasattr(x, 'get_custom_block') else str(x))

    # 第二步：直接在 Python 里调用 MiKTeX 的 pdflatex 编译成 PDF
    print(f"正在调用 MiKTeX 编译 {tex_file}...")
    os.system(f"pdflatex {tex_file}")
    print("PDF 已直接生成！")


if __name__ == '__main__':
    main()