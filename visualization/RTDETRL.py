import sys
import os

# 1. 严格路径处理
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

    # --- 第一部分：Backbone (骨干网络 - HGNetv2/ResNet) ---
    to_Conv("input", 640, 3, offset="(0,0,0)", to="(0,0,0)", height=80, depth=80, width=2, caption="Input"),
    to_ConvRes("bb_p3", 80, 256, offset="(2,0,0)", to="(input-east)", height=45, depth=45, width=6,
               caption="Backbone P3"),
    to_ConvRes("bb_p4", 40, 512, offset="(2,0,0)", to="(bb_p3-east)", height=30, depth=30, width=10,
               caption="Backbone P4"),
    to_ConvRes("bb_p5", 20, 1024, offset="(2,0,0)", to="(bb_p4-east)", height=15, depth=15, width=14,
               caption="Backbone P5"),

    # --- 第二部分：Hybrid Encoder (混合编码器 - 核心创新) ---
    # 这里加宽 width，体现 AIFI 和 CCFF 的复杂计算
    to_ConvRes("encoder", 20, 256, offset="(3,0,0)", to="(bb_p5-east)", height=35, depth=35, width=20,
               caption="Hybrid Encoder"),

    # --- 第三部分：Transformer Decoder (解码器) ---
    # Transformer 结构通常用长条形表示，体现 Query 的并行处理
    to_Conv("decoder", 1, 300, offset="(3,0,0)", to="(encoder-east)", height=40, depth=40, width=8, caption="Decoder"),

    # --- 第四部分：Detection Head (检测头) ---
    # 最终输出位置和类别
    to_Conv("head_box", 1, 4, offset="(3, 1.5, 0)", to="(decoder-east)", height=10, depth=10, width=4,
            caption="BBox Head"),
    to_Conv("head_cls", 1, 80, offset="(3, -1.5, 0)", to="(decoder-east)", height=10, depth=10, width=4,
            caption="Class Head"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    tex_file = namefile + '.tex'

    with open(tex_file, 'w') as f:
        for x in arch:
            if isinstance(x, str):
                f.write(x)
            else:
                f.write(x.get_custom_block() if hasattr(x, 'get_custom_block') else str(x))

    # 自动执行 pdflatex 编译
    print(f"正在编译 RT-DETR-L 架构图...")
    os.system(f"pdflatex {tex_file}")
    print("PDF 已生成，乐哥你过目！")


if __name__ == '__main__':
    main()