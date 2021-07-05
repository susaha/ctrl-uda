import numpy as np
from PIL import Image

palette_17 = [128, 64, 128, # road
            244, 35, 232,   # sidewalk
            70, 70, 70,     # building
            102, 102, 156,  # wall
            190, 153, 153,  # fence
            153, 153, 153,  # pole
            250, 170, 30,   # light
            220, 220, 0,    # sign
            107, 142, 35,   # vegetation
            70, 130, 180,   # sky
            220, 20, 60,    # person
            255, 0, 0,      # rider
            0, 0, 142,      # car
            0, 60, 100,     # bus
            0, 0, 230,      # motocycle
            119, 11, 32,    # bicycle
              128, 128, 128]      # low confiecence pixel set to 0

palette_10 = [128, 64, 128, # road
            # 244, 35, 232,   # sidewalk
            70, 70, 70,     # building
            # 102, 102, 156,  # wall
            # 190, 153, 153,  # fence
            153, 153, 153,  # pole
            250, 170, 30,   # light
            220, 220, 0,    # sign
            107, 142, 35,   # vegetation
            152, 251, 152,  # terrain #  # 244, 35, 232,
            70, 130, 180,   # sky
            # 220, 20, 60,    # person
            # 255, 0, 0,      # rider
            0, 0, 142,      # car
            # 0, 60, 100,     # bus
            # 0, 0, 230,      # motocycle
            # 119, 11, 32]    # bicycle
             0,  0, 70,  # truck  #  102, 102, 156,
            ]
zero_pad = 256 * 3 - len(palette_10)
for i in range(zero_pad):
    palette_10.append(0)


palette_16 = [128, 64, 128, # road          # 804080ff
            244, 35, 232,   # sidewalk     # f423e8ff
            70, 70, 70,     # building      : 464646ff
            102, 102, 156,  # wall      : 666699ff
            190, 153, 153,  # fence         : be9999ff
            153, 153, 153,  # pole      : 999999ff
            250, 170, 30,   # light : faaa1eff
            220, 220, 0,    # sign  : dcdc00ff
            107, 142, 35,   # vegetation    : 6b8e23ff
            70, 130, 180,   # sky   : 4682b4ff
            220, 20, 60,    # person        : dc143cff
            255, 0, 0,      # rider : ff0000ff
            0, 0, 142,      # car       : 00008eff
            0, 60, 100,     # bus   : 003c64ff
            0, 0, 230,      # motocycle: 0000e6ff
            119, 11, 32]    # bicycle : 770b20ff

zero_pad = 256 * 3 - len(palette_16)
for i in range(zero_pad):
    palette_16.append(0)

palette_7 = [128, 64, 128,              # flat
            70, 70, 70,                 # construction
            153, 153, 153,              # object
            107, 142, 35,               # nature
            70, 130, 180,               # sky
            220, 20, 60,                # human
            0, 0, 142]                  # vehicle

zero_pad = 256 * 3 - len(palette_7)
for i in range(zero_pad):
    palette_7.append(0)


palette_5 = [128, 64, 128,
             244, 35, 232,
             102, 102, 156,
             250, 170, 30,
             220, 220, 0]
zero_pad = 256 * 3 - len(palette_5)
for i in range(zero_pad):
    palette_5.append(0)

palette_15 = [128, 64, 128, # road
            244, 35, 232,   # sidewalk
            70, 70, 70,     # building
            102, 102, 156,  # wall
            190, 153, 153,  # fence
            153, 153, 153,  # pole
            250, 170, 30,   # light
            220, 220, 0,    # sign
            107, 142, 35,   # vegetation
            70, 130, 180,   # sky
            220, 20, 60,    # person
            255, 0, 0,      # rider
            0, 0, 142,      # car
            0, 60, 100,     # bus
            0, 0, 230,      # motocycle
            ]


def colorize_mask(num_classes, mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    if num_classes == 16:
        new_mask.putpalette(palette_16)
    elif num_classes == 7:
        new_mask.putpalette(palette_7)

    elif num_classes == 5:
        new_mask.putpalette(palette_5)

    elif num_classes == 10:
        new_mask.putpalette(palette_10)

    elif num_classes == 15:
        new_mask.putpalette(palette_15)

    elif num_classes == 17:
        new_mask.putpalette(palette_17)
    return new_mask