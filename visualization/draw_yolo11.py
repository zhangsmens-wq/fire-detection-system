import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, ".."))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from pycore.tikzeng import *
from pycore.blocks  import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Backbone
    to_Conv("stem_conv", 320, 64, offset="(0,0,0)", to="(0,0,0)", height=80, depth=80, width=4, caption="Stem Conv"),
    to_ConvRes("stage1_c3k2", 160, 128, offset="(1.8,0,0)", to="(stem_conv-east)", height=60, depth=60, width=8, caption="C3k2"),
    to_ConvRes("stage2_c3k2", 80, 256, offset="(1.8,0,0)", to="(stage1_c3k2-east)", height=45, depth=45, width=12, caption="C3k2 (P3)"),
    to_ConvRes("stage3_c3k2", 40, 512, offset="(1.8,0,0)", to="(stage2_c3k2-east)", height=30, depth=30, width=16, caption="C3k2 (P4)"),
    # 修复处：删掉了这里的 color="orange"
    to_ConvRes("stage4_sppf", 20, 1024, offset="(1.8,0,0)", to="(stage3_c3k2-east)", height=20, depth=20, width=20, caption="C3k2+SPPF"),

    # Neck
    # 修复处：删掉了这里的 color="SkyBlue"
    to_ConvRes("neck_up4", 40, 512, offset="(2.5,1.5,0)", to="(stage4_sppf-east)", height=30, depth=30, width=16, caption="Neck Up (P4)"),
    to_ConvRes("neck_up3", 80, 256, offset="(2.5,1.5,0)", to="(neck_up4-east)", height=45, depth=45, width=12, caption="Neck Up (P3)"),

    # Head
    to_Conv("head_out3", 80, "S", offset="(3,0,1)", to="(neck_up3-east)", height=45, depth=45, width=6, caption="Detect (S)"),
    to_Conv("head_out4", 40, "M", offset="(3,0,0)", to="(neck_up4-east)", height=30, depth=30, width=6, caption="Detect (M)"),
    to_Conv("head_out5", 20, "L", offset="(3,0,-1)", to="(stage4_sppf-east)", height=20, depth=20, width=6, caption="Detect (L)"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    with open(namefile + '.tex', 'w') as f:
        for x in arch:
            if isinstance(x, str):
                f.write(x)
            else:
                f.write(x.get_custom_block() if hasattr(x, 'get_custom_block') else str(x))

if __name__ == '__main__':
    main()