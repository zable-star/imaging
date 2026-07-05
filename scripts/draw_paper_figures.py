from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "artifacts" / "figures"


W, H = 1800, 1050
BG = (246, 248, 251)
INK = (30, 42, 58)
MUTED = (88, 102, 120)
BORDER = (203, 214, 228)
BLUE = (36, 116, 166)
BLUE_FILL = (232, 244, 251)
GREEN = (79, 125, 70)
GREEN_FILL = (237, 246, 234)
ORANGE = (196, 129, 45)
ORANGE_FILL = (255, 242, 221)
GRAY_FILL = (238, 242, 246)
WHITE = (255, 255, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_LABEL = font(20, True)
F_BODY = font(16)
F_SMALL = font(13)
F_CODE = font(15)
F_CODE_L = font(16)


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=BORDER, width=2, radius=16):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy, s, fill=INK, fnt=F_BODY, anchor=None):
    draw.text(xy, s, fill=fill, font=fnt, anchor=anchor)


def arrow(draw: ImageDraw.ImageDraw, start, end, fill=(47, 59, 82), width=4):
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 16
    base_x, base_y = x2 - ux * size, y2 - uy * size
    points = [
        (x2, y2),
        (base_x + px * size * 0.45, base_y + py * size * 0.45),
        (base_x - px * size * 0.45, base_y - py * size * 0.45),
    ]
    draw.polygon(points, fill=fill)


def curve_arrow(draw: ImageDraw.ImageDraw, points, fill=(47, 59, 82), width=4):
    draw.line(points, fill=fill, width=width, joint="curve")
    arrow(draw, points[-2], points[-1], fill=fill, width=width)


def feature_bar(draw, x, y, colors, label):
    rounded(draw, (x, y, x + 190, y + 56), (249, 251, 253), (131, 146, 167), 2, 8)
    for idx, color in enumerate(colors):
        draw.rounded_rectangle((x + 18 + idx * 36, y + 15, x + 46 + idx * 36, y + 41), 4, fill=color)
    text(draw, (x + 94, y + 92), label, fnt=F_CODE_L, anchor="mm")


def draw_slice(draw, x, y, name, accent, shape):
    rounded(draw, (x, y, x + 170, y + 145), WHITE, BORDER, 2, 12)
    draw.rounded_rectangle((x + 24, y + 26, x + 146, y + 104), 8, fill=accent[0], outline=accent[1], width=2)
    if shape == 0:
        draw.pieslice((x + 45, y + 42, x + 112, y + 120), 190, 350, fill=(49, 68, 95))
        draw.polygon([(x + 58, y + 86), (x + 108, y + 82), (x + 126, y + 93), (x + 48, y + 95)], fill=(49, 68, 95))
    elif shape == 1:
        draw.polygon([(x + 42, y + 84), (x + 64, y + 52), (x + 96, y + 52), (x + 126, y + 88), (x + 94, y + 74), (x + 64, y + 74)], fill=(95, 69, 37))
    else:
        draw.pieslice((x + 40, y + 36, x + 128, y + 116), 205, 25, fill=(44, 87, 52))
        draw.polygon([(x + 44, y + 88), (x + 68, y + 99), (x + 124, y + 74), (x + 110, y + 61)], fill=(44, 87, 52))
    text(draw, (x + 85, y + 128), name, fnt=F_BODY, anchor="mm")


def draw_compact_slice(draw, x, y, name, accent, shape):
    draw.rounded_rectangle((x, y, x + 138, y + 72), 8, fill=accent[0], outline=accent[1], width=2)
    if shape == 0:
        draw.pieslice((x + 28, y + 14, x + 88, y + 70), 190, 350, fill=(49, 68, 95))
        draw.polygon([(x + 42, y + 50), (x + 96, y + 48), (x + 116, y + 58), (x + 30, y + 60)], fill=(49, 68, 95))
    elif shape == 1:
        draw.polygon([(x + 24, y + 52), (x + 48, y + 25), (x + 82, y + 25), (x + 112, y + 56), (x + 80, y + 45), (x + 48, y + 45)], fill=(95, 69, 37))
    else:
        draw.pieslice((x + 28, y + 12, x + 116, y + 84), 205, 25, fill=(44, 87, 52))
        draw.polygon([(x + 26, y + 56), (x + 58, y + 68), (x + 118, y + 46), (x + 95, y + 35)], fill=(44, 87, 52))
    text(draw, (x + 69, y + 95), name, MUTED, F_SMALL, anchor="mm")


def draw_method_architecture():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    text(draw, (90, 56), "Gated Slice Attention-Residual Network", fnt=F_TITLE)
    text(draw, (90, 92), "Shared slice encoding, interpretable attention weights, and concat residual compensation", MUTED)
    rounded(draw, (70, 140, 1730, 965), WHITE, BORDER, 2, 18)

    text(draw, (120, 178), "Input gated slices", fnt=F_LABEL)
    text(draw, (120, 211), "One object sample: [S=3, C=1, H, W]", MUTED)
    draw_slice(draw, 120, 270, "gate_0", ((226, 241, 250), (141, 182, 210)), 0)
    draw_slice(draw, 120, 445, "gate_1", ((255, 242, 221), (199, 153, 62)), 1)
    draw_slice(draw, 120, 620, "gate_2", ((229, 245, 226), (118, 169, 115)), 2)

    rounded(draw, (380, 355, 625, 625), (237, 243, 248), (196, 208, 220), 2, 14)
    text(draw, (502, 400), "Shared", fnt=F_LABEL, anchor="mm")
    text(draw, (502, 430), "SliceEncoder", fnt=F_LABEL, anchor="mm")
    for idx, line in enumerate(["CNN blocks", "BatchNorm", "ReLU", "Global pooling"]):
        text(draw, (502, 474 + idx * 30), line, MUTED, anchor="mm")
    text(draw, (502, 600), "f_i in R^128", fnt=F_CODE_L, anchor="mm")
    arrow(draw, (290, 342), (370, 405))
    arrow(draw, (290, 518), (370, 490))
    arrow(draw, (290, 692), (370, 575))

    text(draw, (715, 178), "Per-slice features", fnt=F_LABEL)
    feature_bar(draw, 700, 270, [(183, 215, 238), (213, 231, 242), (142, 190, 222), (199, 220, 235)], "f_0")
    feature_bar(draw, 700, 445, [(242, 200, 121), (247, 223, 165), (217, 169, 74), (237, 211, 153)], "f_1")
    feature_bar(draw, 700, 620, [(141, 199, 138), (185, 220, 180), (105, 163, 101), (211, 233, 207)], "f_2")
    curve_arrow(draw, [(625, 490), (650, 330), (690, 300)])
    arrow(draw, (625, 490), (690, 475))
    curve_arrow(draw, [(625, 490), (650, 650), (690, 650)])

    rounded(draw, (1000, 205, 1280, 375), BLUE_FILL, (105, 168, 205), 2, 14)
    text(draw, (1140, 244), "Attention scorer", fnt=F_LABEL, anchor="mm")
    text(draw, (1140, 282), "MLP + softmax over S", MUTED, anchor="mm")
    text(draw, (1140, 324), "alpha_0, alpha_1, alpha_2", fnt=F_CODE_L, anchor="mm")
    rounded(draw, (1340, 225, 1575, 355), BLUE_FILL, (105, 168, 205), 2, 14)
    text(draw, (1458, 272), "Weighted sum", fnt=F_LABEL, anchor="mm")
    text(draw, (1458, 313), "f_att = sum alpha_i f_i", fnt=F_CODE_L, anchor="mm")
    curve_arrow(draw, [(890, 298), (930, 250), (990, 250)], BLUE)
    curve_arrow(draw, [(890, 475), (930, 350), (990, 300)], BLUE)
    curve_arrow(draw, [(890, 650), (930, 405), (990, 338)], BLUE)
    arrow(draw, (1280, 290), (1330, 290), BLUE)

    rounded(draw, (1000, 535, 1280, 705), GREEN_FILL, (138, 178, 127), 2, 14)
    text(draw, (1140, 574), "Concat residual", fnt=F_LABEL, anchor="mm")
    text(draw, (1140, 618), "[f_0, f_1, f_2]", fnt=F_CODE_L, anchor="mm")
    text(draw, (1140, 656), "LayerNorm + MLP", MUTED, anchor="mm")
    text(draw, (1140, 690), "f_res", fnt=F_CODE_L, anchor="mm")
    curve_arrow(draw, [(890, 298), (945, 455), (990, 590)], GREEN)
    curve_arrow(draw, [(890, 475), (935, 510), (990, 610)], GREEN)
    curve_arrow(draw, [(890, 650), (925, 655), (990, 650)], GREEN)

    draw.ellipse((1414, 461, 1502, 549), fill=WHITE, outline=(85, 98, 116), width=3)
    text(draw, (1458, 504), "+", fnt=font(44, True), anchor="mm")
    text(draw, (1458, 575), "residual add", MUTED, anchor="mm")
    arrow(draw, (1458, 355), (1458, 451), BLUE)
    curve_arrow(draw, [(1280, 620), (1365, 620), (1428, 540)], GREEN)

    rounded(draw, (1575, 430, 1705, 580), ORANGE_FILL, (217, 155, 67), 2, 14)
    text(draw, (1640, 480), "Classifier", fnt=F_LABEL, anchor="mm")
    text(draw, (1640, 520), "LayerNorm", MUTED, anchor="mm")
    text(draw, (1640, 548), "MLP", MUTED, anchor="mm")
    text(draw, (1640, 576), "K=6", fnt=F_CODE_L, anchor="mm")
    arrow(draw, (1502, 505), (1565, 505))

    rounded(draw, (1515, 660, 1690, 850), (249, 251, 253), BORDER, 2, 10)
    text(draw, (1602, 702), "Outputs", fnt=F_LABEL, anchor="mm")
    for idx, line in enumerate(["class prediction", "attention weights", "slice features"]):
        text(draw, (1602, 742 + idx * 32), line, MUTED, anchor="mm")
    draw.rounded_rectangle((1540, 820, 1574, 832), 3, fill=(124, 182, 217))
    draw.rounded_rectangle((1580, 820, 1638, 832), 3, fill=(240, 189, 100))
    draw.rounded_rectangle((1644, 820, 1667, 832), 3, fill=(126, 185, 120))
    arrow(draw, (1640, 580), (1640, 650))

    rounded(draw, (104, 845, 744, 919), (251, 252, 254), (214, 222, 232), 2, 14)
    text(draw, (130, 870), "Main idea: attention keeps interpretable gate-level weights;", MUTED)
    text(draw, (130, 895), "the residual branch restores slice-specific information preserved by concat.", MUTED)
    rounded(draw, (790, 845, 1405, 919), (251, 252, 254), (214, 222, 232), 2, 14)
    text(draw, (815, 870), "Recommended paper model: attention-residual.", MUTED)
    text(draw, (815, 895), "Concat is reported as a high-accuracy empirical baseline.", MUTED)

    out = FIGURE_DIR / "method_architecture_attention_residual.png"
    img.save(out, quality=95)
    return out


def draw_rendering_pipeline():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    text(draw, (90, 56), "Physically Inspired Gated Slice Rendering Pipeline", fnt=F_TITLE)
    text(draw, (90, 92), "From ModelNet geometry to three range-gated grayscale slices and the image2d abnormal class", MUTED)
    rounded(draw, (70, 140, 1730, 965), WHITE, BORDER, 2, 18)

    rounded(draw, (120, 205, 365, 410), WHITE, BORDER, 2, 14)
    text(draw, (242, 244), "3D model input", fnt=F_LABEL, anchor="mm")
    text(draw, (242, 282), "ModelNet10 geometry", MUTED, anchor="mm")
    text(draw, (242, 318), "OFF -> OBJ", fnt=F_CODE_L, anchor="mm")
    draw.polygon([(204, 402), (249, 370), (314, 394), (268, 430)], fill=(215, 226, 236), outline=(123, 135, 152))
    draw.polygon([(249, 370), (282, 354), (344, 377), (314, 394)], fill=(238, 243, 248), outline=(123, 135, 152))
    draw.polygon([(314, 394), (344, 377), (296, 418), (268, 430)], fill=(189, 201, 214), outline=(123, 135, 152))

    rounded(draw, (465, 205, 895, 570), BLUE_FILL, (105, 168, 205), 2, 14)
    text(draw, (680, 244), "Blender gated-view scene", fnt=F_LABEL, anchor="mm")
    text(draw, (680, 282), "Camera-axis depth is used as range R", MUTED, anchor="mm")
    draw.rounded_rectangle((520, 325, 835, 510), 14, fill=(231, 239, 246), outline=(156, 175, 194), width=2)
    draw.polygon([(566, 461), (602, 392), (650, 443), (690, 367), (780, 461)], fill=(184, 200, 216))
    draw.pieslice((560, 380, 690, 520), 180, 360, fill=(66, 85, 109))
    draw.pieslice((665, 365, 800, 520), 180, 360, fill=(46, 76, 58))
    draw.ellipse((557, 339, 589, 371), fill=(240, 189, 100))
    draw.line([(552, 354), (490, 332)], fill=(224, 170, 77), width=4)
    draw.line([(552, 360), (490, 388)], fill=(224, 170, 77), width=4)
    draw.rounded_rectangle((782, 357, 810, 427), 4, fill=(57, 70, 90))
    draw.polygon([(782, 392), (726, 360), (726, 424)], fill=(165, 176, 190))
    text(draw, (677, 536), "simplified illumination, surface normal, and range attenuation", MUTED, F_SMALL, anchor="mm")

    rounded(draw, (1010, 205, 1355, 570), ORANGE_FILL, (217, 155, 67), 2, 14)
    text(draw, (1182, 244), "Range-gate response", fnt=F_LABEL, anchor="mm")
    text(draw, (1182, 282), "Pulse-window overlap in range domain", MUTED, anchor="mm")
    draw.line([(1065, 472), (1305, 472)], fill=(102, 112, 133), width=2)
    draw.line([(1065, 472), (1065, 335)], fill=(102, 112, 133), width=2)
    draw.line([(1075, 472), (1118, 472), (1160, 370), (1202, 370), (1244, 472), (1300, 472)], fill=BLUE, width=4)
    draw.line([(1105, 472), (1142, 472), (1186, 345), (1230, 345), (1274, 472), (1300, 472)], fill=ORANGE, width=4)
    draw.line([(1138, 472), (1182, 472), (1222, 392), (1262, 392), (1300, 472)], fill=GREEN, width=4)
    text(draw, (1070, 506), "R", MUTED, F_SMALL)
    text(draw, (1038, 344), "W_g(R)", MUTED, F_SMALL)
    text(draw, (1182, 535), "I(R) proportional to W_g(R)", fnt=F_CODE, anchor="mm")

    rounded(draw, (1450, 205, 1680, 650), GREEN_FILL, (138, 178, 127), 2, 14)
    text(draw, (1565, 244), "Gated slices", fnt=F_LABEL, anchor="mm")
    text(draw, (1565, 282), "3 grayscale PNGs", MUTED, anchor="mm")
    draw_compact_slice(draw, 1495, 320, "gate_0", ((226, 241, 250), (125, 184, 220)), 0)
    draw_compact_slice(draw, 1495, 440, "gate_1", ((255, 242, 221), (217, 155, 67)), 1)
    draw_compact_slice(draw, 1495, 560, "gate_2", ((229, 245, 226), (117, 179, 110)), 2)

    arrow(draw, (365, 308), (455, 308))
    arrow(draw, (895, 388), (1000, 388))
    arrow(draw, (1355, 388), (1440, 388))

    rounded(draw, (475, 645, 1365, 760), GRAY_FILL, (198, 208, 220), 2, 14)
    text(draw, (515, 680), "Simplified gated intensity model", fnt=F_LABEL)
    text(draw, (515, 718), "I(R) = gate response x Lambert term x range loss x atmospheric transmission", fnt=F_CODE)
    text(draw, (515, 748), "Physics-inspired rendering baseline; not a full hardware-level laser imaging simulation.", MUTED, F_SMALL)

    rounded(draw, (120, 795, 610, 915), WHITE, BORDER, 2, 14)
    text(draw, (150, 828), "Normal 3D classes", fnt=F_LABEL)
    text(draw, (150, 866), "chair, desk, sofa, bed, toilet", MUTED)
    text(draw, (150, 892), "Each sample keeps three real gated slices.", MUTED, F_SMALL)
    rounded(draw, (690, 795, 1145, 915), WHITE, BORDER, 2, 14)
    text(draw, (720, 828), "Abnormal image2d class", fnt=F_LABEL)
    text(draw, (720, 866), "one informative slice + two black slices", MUTED)
    text(draw, (720, 892), "Used to test 2D-degenerate input recognition.", MUTED, F_SMALL)
    rounded(draw, (1230, 795, 1680, 915), ORANGE_FILL, (217, 155, 67), 2, 14)
    text(draw, (1260, 828), "Six-class gated-slice dataset", fnt=F_LABEL)
    text(draw, (1260, 866), "[gate_0, gate_1, gate_2] per sample", MUTED)
    text(draw, (1260, 892), "Input to attention-residual classifier.", MUTED, F_SMALL)
    curve_arrow(draw, [(1565, 650), (1565, 730), (1368, 790)], BLUE)
    arrow(draw, (610, 848), (680, 848))
    arrow(draw, (1145, 848), (1220, 848))

    for idx, fill in enumerate([(255, 242, 221), (16, 21, 28), (16, 21, 28)]):
        draw.rounded_rectangle((965 + idx * 84, 604, 1037 + idx * 84, 648), 5, fill=fill, outline=(85, 98, 116), width=2)
    text(draw, (1001, 674), "image", MUTED, F_SMALL, anchor="mm")
    text(draw, (1085, 674), "black", MUTED, F_SMALL, anchor="mm")
    text(draw, (1169, 674), "black", MUTED, F_SMALL, anchor="mm")

    out = FIGURE_DIR / "gated_slice_rendering_pipeline.png"
    img.save(out, quality=95)
    return out


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in [draw_method_architecture(), draw_rendering_pipeline()]:
        print(path)


if __name__ == "__main__":
    main()
