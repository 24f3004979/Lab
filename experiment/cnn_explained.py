"""
cnn_explained.py  -  End-to-end CNN explainer animation (Manim Community Edition)
Automated AI generated code 
What it does (all in ONE script):
  1. Downloads MNIST (handwritten digits) and trains a tiny CNN on CPU (~1-2 min, cached afterwards).
     While training it records snapshots of the conv-1 filters and the loss curve.
  2. Picks one real handwritten test digit and runs it through the trained network,
     capturing every intermediate tensor.
  3. Renders a video that walks through:
        raw pixels -> filters learning (random -> edge detectors) -> sliding convolution
        -> ReLU -> max-pooling -> second conv layer -> flatten -> fully connected -> softmax -> prediction

Install:
    pip install manim torch torchvision numpy
    (no LaTeX needed - only Text() is used)

Render:
    manim -pqm cnn_explained.py CNNExplained          # 720p preview
    manim -pqh cnn_explained.py CNNExplained          # 1080p
    CNN_DIGIT=3 manim -pqm cnn_explained.py CNNExplained   # explain a different digit (0-9)

The trained weights are cached in cnn_cache.pkl next to this file. Delete it to retrain.
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from manim import *  # noqa: F401,F403

import numpy as np  # re-import in case the star import shadowed it

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cnn_cache.pkl")
TARGET_DIGIT = int(os.environ.get("CNN_DIGIT", "7"))


# =====================================================================================
#  PART 1 - MODEL, TRAINING, DATA
# =====================================================================================
class TinyCNN(nn.Module):
    """28x28 -> conv(4,3x3) -> relu -> pool -> conv(8,3x3) -> relu -> pool -> fc(200->10)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)   # 28 -> 26
        self.conv2 = nn.Conv2d(4, 8, 3)   # 13 -> 11
        self.fc = nn.Linear(8 * 5 * 5, 10)  # 5x5x8 = 200

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.flatten(1))


def load_mnist():
    from torchvision import datasets
    root = os.path.join(HERE, "mnist_data")
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    prep = lambda d: (d.data.float().div(255).unsqueeze(1), d.targets)
    return prep(tr), prep(te)


def train_model(epochs=2, batch=128):
    torch.manual_seed(0)
    np.random.seed(0)
    (xtr, ytr), (xte, yte) = load_mnist()
    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), 1e-3)

    n = len(xtr)
    steps_per = n // batch
    total = epochs * steps_per
    snap_at = {0, 20, 80, 250}
    snaps, hist_steps, hist_loss = {}, [], []
    ema, step = None, 0

    print(f"[train] {total} steps ...")
    for _ in range(epochs):
        perm = torch.randperm(n)
        for b in range(steps_per):
            if step in snap_at:  # filters BEFORE this update
                snaps[step] = model.conv1.weight.detach().clone().numpy()[:, 0]
            idx = perm[b * batch:(b + 1) * batch]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            l = loss.item()
            ema = l if ema is None else 0.9 * ema + 0.1 * l
            if step % 10 == 0:
                hist_steps.append(step)
                hist_loss.append(ema)
            step += 1
    snaps[total] = model.conv1.weight.detach().clone().numpy()[:, 0]  # final filters

    model.eval()
    with torch.no_grad():
        acc = (model(xte).argmax(1) == yte).float().mean().item()
    print(f"[train] test accuracy = {acc:.4f}")

    return {
        "state": {k: v.numpy() for k, v in model.state_dict().items()},
        "snaps": snaps,
        "hist_steps": hist_steps,
        "hist_loss": hist_loss,
        "acc": acc,
        "test_x": xte[:500, 0].numpy(),
        "test_y": yte[:500].numpy(),
    }


def build_model(state):
    m = TinyCNN()
    m.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
    m.eval()
    return m


def trace(model, img):
    """Run one image through the network and return every intermediate result as numpy."""
    x = torch.tensor(img, dtype=torch.float32)[None, None]
    with torch.no_grad():
        c1 = model.conv1(x)
        r1 = F.relu(c1)
        p1 = F.max_pool2d(r1, 2)
        c2 = model.conv2(p1)
        r2 = F.relu(c2)
        p2 = F.max_pool2d(r2, 2)
        flat = p2.flatten(1)
        logits = model.fc(flat)
        probs = F.softmax(logits, 1)
    g = lambda t: t[0].numpy()
    return dict(
        c1=g(c1), r1=g(r1), p1=g(p1), c2=g(c2), r2=g(r2), p2=g(p2),
        flat=flat[0].numpy(), logits=logits[0].numpy(), probs=probs[0].numpy(),
        w1=model.conv1.weight[:, 0].detach().numpy(),
        b1=model.conv1.bias.detach().numpy(),
        w2=model.conv2.weight.detach().numpy(),
    )


def get_data():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            D = pickle.load(f)
    else:
        D = train_model()
        with open(CACHE, "wb") as f:
            pickle.dump(D, f)

    model = build_model(D["state"])
    # choose the first correctly-classified test sample of the requested digit
    chosen = None
    for i in np.where(D["test_y"] == TARGET_DIGIT)[0]:
        T = trace(model, D["test_x"][i])
        if int(T["probs"].argmax()) == TARGET_DIGIT:
            chosen = (i, T)
            break
    if chosen is None:
        i = 0
        chosen = (i, trace(model, D["test_x"][i]))
    i, T = chosen
    D["img"] = D["test_x"][i]
    D["label"] = int(D["test_y"][i])
    D["T"] = T
    return D


# =====================================================================================
#  PART 2 - DRAWING HELPERS
# =====================================================================================
def gray_arr(a, lo, hi):
    t = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
    v = (t * 255).astype(np.uint8)
    return np.stack([v, v, v], -1)


def div_arr(a, m):
    """Diverging map: red = positive, white = 0, blue = negative."""
    t = np.clip(a / (m + 1e-8), -1, 1)
    pos, neg = np.clip(t, 0, 1), np.clip(-t, 0, 1)
    r, g, b = 1 - neg, 1 - np.maximum(pos, neg), 1 - pos
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(x) for x in rgb)


def to_img(rgb, height):
    im = ImageMobject(rgb)
    im.set_height(height)
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])  # crisp pixels
    return im


def number_grid(vals, cell, fmt, rgbfn, font=18):
    rows, cols = vals.shape
    cells = []
    for i in range(rows):
        for j in range(cols):
            v = float(vals[i, j])
            rgb = rgbfn(v)
            sq = Square(cell, stroke_width=1, stroke_color=GREY_B)
            sq.set_fill(hexc(rgb), 1)
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            t = Text(fmt.format(v), font_size=font, color=BLACK if lum > 140 else WHITE)
            if t.width > cell * 0.92:
                t.scale_to_fit_width(cell * 0.92)
            t.move_to(sq)
            cells.append(VGroup(sq, t))
    return VGroup(*cells).arrange_in_grid(rows, cols, buff=0)


def cell_center(mob, r, c, nrows, ncols):
    pw, ph = mob.width / ncols, mob.height / nrows
    return np.array([mob.get_left()[0] + (c + 0.5) * pw,
                     mob.get_top()[1] - (r + 0.5) * ph, 0.0])


def map_item(arr, mode, height, label=None):
    if mode == "div":
        rgb = div_arr(arr, np.abs(arr).max())
    else:
        rgb = gray_arr(arr, 0, max(float(arr.max()), 1e-6))
    im = to_img(rgb, height)
    fr = SurroundingRectangle(im, buff=0, color=GREY_B, stroke_width=1.5)
    parts = [im, fr]
    if label:
        parts.append(Text(label, font_size=18, color=GREY_B).next_to(im, DOWN, buff=0.08))
    return Group(*parts)


# =====================================================================================
#  PART 3 - THE ANIMATION
# =====================================================================================
class CNNExplained(Scene):
    # ---------- small utilities ----------
    def set_title(self, text):
        new = Text(text, font_size=32, weight=BOLD).to_corner(UL, buff=0.35)
        if self._title is None:
            self.play(FadeIn(new, shift=DOWN * 0.2), run_time=0.6)
        else:
            self.play(ReplacementTransform(self._title, new), run_time=0.6)
        self._title = new

    def say(self, text):
        new = Text(text, font_size=24, color=GREY_A).to_edge(DOWN, buff=0.3)
        if self._cap is None:
            self.play(FadeIn(new), run_time=0.5)
        else:
            self.play(ReplacementTransform(self._cap, new), run_time=0.5)
        self._cap = new

    def clear_stage(self):
        others = [m for m in self.mobjects if m is not self._title and m is not self._cap]
        if others:
            self.play(*[FadeOut(m) for m in others], run_time=0.7)

    # ---------- main ----------
    def construct(self):
        self._title, self._cap = None, None
        D = get_data()
        T = D["T"]
        self.D, self.T = D, T
        self.f = int(np.argmax(np.var(T["r1"].reshape(4, -1), axis=1)))  # most "active" filter

        self.sec_intro()
        self.sec_pixels()
        self.sec_training()
        self.sec_convolution()
        self.sec_relu_pool()
        self.sec_conv2()
        self.sec_classifier()
        self.sec_summary()

    # ---------- 0. intro ----------
    def sec_intro(self):
        D = self.D
        title = Text("How a CNN Sees a Digit", font_size=56, weight=BOLD)
        sub = Text("Training, filters, feature maps and prediction - end to end",
                   font_size=26, color=GREY_B).next_to(title, DOWN, buff=0.3)
        pic = to_img(gray_arr(D["img"], 0, 1), 1.8).next_to(sub, DOWN, buff=0.6)
        pic_fr = SurroundingRectangle(pic, buff=0, color=WHITE, stroke_width=2)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.2), FadeIn(pic), Create(pic_fr))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(sub), FadeOut(pic), FadeOut(pic_fr))

    # ---------- 1. pixels ----------
    def sec_pixels(self):
        D = self.D
        self.set_title("1 | An image is just numbers")
        img = to_img(gray_arr(D["img"], 0, 1), 4.4).move_to(LEFT * 4.2 + DOWN * 0.1)
        fr = SurroundingRectangle(img, buff=0, color=WHITE, stroke_width=2)
        self.play(FadeIn(img), Create(fr))
        self.say("A handwritten MNIST digit: a 28x28 grid of brightness values")
        self.wait(1)

        # most "interesting" 5x5 patch (highest variance = stroke edge)
        a = D["img"]
        best, pr, pc = -1, 0, 0
        for r in range(0, 24):
            for c in range(0, 24):
                s = a[r:r + 5, c:c + 5].std()
                if s > best:
                    best, pr, pc = s, r, c
        self.pr, self.pc = pr, pc

        cA = cell_center(img, pr, pc, 28, 28)
        cB = cell_center(img, pr + 4, pc + 4, 28, 28)
        rect = Rectangle(width=5 * img.width / 28, height=5 * img.height / 28,
                         color=YELLOW, stroke_width=3).move_to((cA + cB) / 2)
        patch = (a[pr:pr + 5, pc:pc + 5] * 255).round()
        grid = number_grid(patch, 0.8, "{:.0f}", lambda v: gray_arr(np.float32(v), 0, 255), 22)
        grid.move_to(RIGHT * 3.2 + DOWN * 0.1)
        l1 = Line(rect.get_corner(UR), grid.get_corner(UL), color=YELLOW, stroke_opacity=0.6)
        l2 = Line(rect.get_corner(DR), grid.get_corner(DL), color=YELLOW, stroke_opacity=0.6)
        self.play(Create(rect))
        self.play(Create(l1), Create(l2), FadeIn(grid))
        self.say("Zoom in: 0 = black background, 255 = bright pen stroke")
        self.wait(2.5)
        self.clear_stage()

    # ---------- 2. training ----------
    def sec_training(self):
        D = self.D
        self.set_title("2 | Training: filters learn by themselves")
        snap_steps = sorted(D["snaps"].keys())
        hs, hl = np.array(D["hist_steps"]), np.array(D["hist_loss"])

        axes = Axes(x_range=[0, float(hs[-1]), float(hs[-1]) / 4],
                    y_range=[0, float(hl.max()) * 1.1, 0.5],
                    x_length=9, y_length=2.3, tips=False,
                    axis_config={"include_numbers": False, "stroke_width": 2}
                    ).move_to(DOWN * 1.9)
        xl = Text("training steps", font_size=18, color=GREY_B).next_to(axes, DOWN, buff=0.1)
        yl = Text("loss (error)", font_size=18, color=GREY_B).next_to(axes, UP, buff=0.1).align_to(axes, LEFT)
        pts = [axes.c2p(x, y) for x, y in zip(hs, hl)]

        idxs, prev = [], -1
        for s in snap_steps:
            i = int(np.argmin(np.abs(hs - s)))
            i = min(max(i, prev + 1), len(hs) - 1)
            idxs.append(i)
            prev = i

        def filt_row(W):
            items = []
            for i in range(4):
                im = to_img(div_arr(W[i], np.abs(W[i]).max()), 1.5)
                frm = SurroundingRectangle(im, buff=0, color=GREY_B, stroke_width=1.5)
                lb = Text(f"Filter {i + 1}", font_size=18, color=GREY_B).next_to(im, DOWN, buff=0.08)
                items.append(Group(im, frm, lb))
            return Group(*items).arrange(RIGHT, buff=0.7).move_to(UP * 0.9)

        def step_label(s, k):
            if k == 0:
                txt = "Step 0 - random noise (network knows nothing)"
            elif k == len(snap_steps) - 1:
                txt = f"Step {s} - trained filters"
            else:
                txt = f"Step {s}"
            return Text(txt, font_size=26, color=YELLOW).move_to(UP * 2.55)

        row = filt_row(D["snaps"][snap_steps[0]])
        lab = step_label(snap_steps[0], 0)
        dot = Dot(color=RED).move_to(pts[idxs[0]])
        self.play(FadeIn(row), FadeIn(lab), Create(axes), FadeIn(xl), FadeIn(yl), FadeIn(dot))
        self.say("Each 3x3 filter starts as random numbers (red = +, blue = -)")
        self.wait(1.5)
        self.say("Backpropagation nudges every filter weight to lower the loss")

        for k in range(1, len(snap_steps)):
            s = snap_steps[k]
            new_row = filt_row(D["snaps"][s])
            new_lab = step_label(s, k)
            anims = [FadeOut(row), FadeIn(new_row), ReplacementTransform(lab, new_lab),
                     dot.animate.move_to(pts[idxs[k]])]
            if idxs[k] > idxs[k - 1]:
                seg = VMobject(stroke_color=YELLOW, stroke_width=3)
                seg.set_points_as_corners(pts[idxs[k - 1]:idxs[k] + 1])
                anims.append(Create(seg))
            self.play(*anims, run_time=2.0)
            row, lab = new_row, new_lab
            self.wait(0.6)

        acc = Text(f"Test accuracy: {D['acc'] * 100:.1f}%", font_size=28, color=GREEN)
        acc.next_to(axes, UP, buff=0.1).align_to(axes, RIGHT)
        self.play(FadeIn(acc, shift=UP * 0.2))
        self.say("Filters became stroke / edge detectors - nobody programmed them!")
        self.wait(2.5)
        self.clear_stage()

    # ---------- 3. convolution ----------
    def sec_convolution(self):
        D, T, f = self.D, self.T, self.f
        self.set_title("3 | Convolution: sliding a filter over the image")
        H = 4.0
        fm, ker = T["c1"][f], T["w1"][f]

        inp = to_img(gray_arr(D["img"], 0, 1), H).move_to(LEFT * 5.0 + DOWN * 0.3)
        inp_fr = SurroundingRectangle(inp, buff=0, color=WHITE, stroke_width=2)
        out_fr = Square(H, color=WHITE, stroke_width=2).move_to(RIGHT * 5.0 + DOWN * 0.3)
        in_lab = Text("Input 28x28", font_size=20, color=GREY_B).next_to(inp, UP, buff=0.15)
        out_lab = Text("Feature map 26x26", font_size=20, color=GREY_B).next_to(out_fr, UP, buff=0.15)

        mk = np.abs(ker).max()
        kgrid = number_grid(ker, 0.5, "{:+.2f}", lambda v: div_arr(np.float32(v), mk), 15)
        kgrid.move_to([1.3, 1.2, 0])
        k_lab = Text(f"Learned filter {f + 1}", font_size=20, color=GREY_B).next_to(kgrid, UP, buff=0.12)
        times = Text("x", font_size=30).move_to([0.0, 1.2, 0])
        p_lab = Text("Image patch", font_size=20, color=GREY_B).move_to([-1.3, 1.2 + 1.0, 0])

        self.play(FadeIn(inp), Create(inp_fr), Create(out_fr), FadeIn(in_lab), FadeIn(out_lab))
        self.play(FadeIn(kgrid), FadeIn(k_lab), FadeIn(times), FadeIn(p_lab))

        r0 = int(np.clip(self.pr + 1, 0, 23))
        positions = [(r0, int(np.clip(self.pc - 1 + k, 0, 25))) for k in range(6)]
        pw_in, pw_out = inp.width / 28, out_fr.width / 26
        m = float(np.abs(fm).max())

        win = Square(3 * pw_in, color=YELLOW, stroke_width=4)
        mark = Square(pw_out, color=YELLOW, stroke_width=3)
        pgrid, res, pix = None, None, VGroup()

        for n, (r, c) in enumerate(positions):
            patch = D["img"][r:r + 3, c:c + 3]
            newp = number_grid(patch, 0.5, "{:.1f}", lambda v: gray_arr(np.float32(v), 0, 1), 15)
            newp.move_to([-1.3, 1.2, 0])
            val = float(fm[r, c])
            newres = Text(f"sum(patch x filter) + bias = {val:+.2f}", font_size=24).move_to([0, -1.5, 0])
            wpos = cell_center(inp, r + 1, c + 1, 28, 28)
            opos = cell_center(out_fr, r, c, 26, 26)
            sq = Square(pw_out, stroke_width=0).set_fill(hexc(div_arr(np.float32(val), m)), 1).move_to(opos)

            if n == 0:
                win.move_to(wpos)
                mark.move_to(opos)
                pgrid, res = newp, newres
                self.play(FadeIn(win), FadeIn(mark), FadeIn(pgrid), FadeIn(res))
                self.say("Multiply the 3x3 patch by the filter, add everything up -> ONE number")
                self.wait(1.2)
            else:
                self.play(win.animate.move_to(wpos), mark.animate.move_to(opos),
                          Transform(pgrid, newp), Transform(res, newres), run_time=0.9)
            pix.add(sq)
            self.play(FadeIn(sq, scale=1.5), run_time=0.3)
            self.wait(0.3)

        self.say("Slide the SAME filter to every position -> a complete feature map")
        fmimg = to_img(div_arr(fm, m), H).move_to(out_fr)
        self.play(FadeOut(win), FadeOut(mark), FadeOut(pgrid), FadeOut(res),
                  FadeOut(kgrid), FadeOut(k_lab), FadeOut(times), FadeOut(p_lab))
        self.play(FadeIn(fmimg), FadeOut(pix), run_time=1.5)
        note = Text("bright red = this filter fired strongly here\n(it is detecting its pattern)",
                    font_size=22, color=GREY_A).move_to([0, 0.6, 0])
        self.play(FadeIn(note))
        self.wait(3)
        self.clear_stage()

    # ---------- 4. relu + pooling ----------
    def sec_relu_pool(self):
        T, f = self.T, self.f
        self.set_title("4 | ReLU and Max-Pooling")

        c1 = Group(*[map_item(T["c1"][i], "div", 2.4, f"Filter {i + 1}") for i in range(4)])
        c1.arrange(RIGHT, buff=0.6).move_to(UP * 0.2)
        self.play(FadeIn(c1))
        self.say("Conv layer 1 output: 4 feature maps (26x26), one per filter")
        self.wait(2)

        r1 = Group(*[map_item(T["r1"][i], "gray", 2.4, f"Filter {i + 1}") for i in range(4)])
        r1.arrange(RIGHT, buff=0.6).move_to(UP * 0.2)
        self.play(FadeOut(c1), FadeIn(r1), run_time=1.2)
        self.say("ReLU: negative values -> 0, keep only where the pattern was found")
        self.wait(2.5)
        self.clear_stage()

        # --- pooling mini demo on a real 4x4 block ---
        A = T["r1"][f]
        best, br, bc = -1, 0, 0
        for r in range(0, 26 - 3, 2):
            for c in range(0, 26 - 3, 2):
                s = A[r:r + 4, c:c + 4].sum()
                if s > best:
                    best, br, bc = s, r, c
        blk = A[br:br + 4, bc:bc + 4]
        hi = float(max(blk.max(), 1e-6))
        g_in = number_grid(blk, 0.75, "{:.1f}", lambda v: gray_arr(np.float32(v), 0, hi), 18)
        g_in.move_to(LEFT * 3.3 + DOWN * 0.1)
        out = np.array([[blk[0:2, 0:2].max(), blk[0:2, 2:4].max()],
                        [blk[2:4, 0:2].max(), blk[2:4, 2:4].max()]])
        g_out = number_grid(out, 0.9, "{:.1f}", lambda v: gray_arr(np.float32(v), 0, hi), 22)
        g_out.move_to(RIGHT * 3.3 + DOWN * 0.1)
        arrow = Arrow(LEFT * 1.0, RIGHT * 1.0, buff=0).move_to(DOWN * 0.1)
        at = Text("max of each 2x2", font_size=20, color=GREY_B).next_to(arrow, UP, buff=0.1)

        cols = [RED, GREEN, BLUE, ORANGE]
        rects, orects = [], []
        for bi in range(2):
            for bj in range(2):
                cells = VGroup(*[g_in[(2 * bi + di) * 4 + (2 * bj + dj)]
                                 for di in range(2) for dj in range(2)])
                rects.append(SurroundingRectangle(cells, buff=0, color=cols[bi * 2 + bj], stroke_width=5))
                orects.append(SurroundingRectangle(g_out[bi * 2 + bj], buff=0,
                                                   color=cols[bi * 2 + bj], stroke_width=5))
        self.play(FadeIn(g_in))
        self.say("Max-pooling: keep only the strongest response in each 2x2 window")
        self.play(*[Create(r) for r in rects])
        self.play(FadeIn(arrow), FadeIn(at))
        self.play(*[FadeIn(g_out[i]) for i in range(4)], *[Create(o) for o in orects])
        self.wait(2.5)
        self.clear_stage()

        # --- pooled maps ---
        p1 = Group(*[map_item(T["p1"][i], "gray", 2.4, f"Filter {i + 1}") for i in range(4)])
        p1.arrange(RIGHT, buff=0.6).move_to(UP * 0.2)
        self.play(FadeIn(p1))
        self.say("Pooled maps are 13x13: half the size, same important features")
        self.wait(3)
        self.clear_stage()

    # ---------- 5. second conv layer ----------
    def sec_conv2(self):
        T = self.T
        self.set_title("5 | Layer 2: combining edges into shapes")
        W = T["w2"]  # (8, 4, 3, 3)

        rows = []
        for j in range(8):
            m = float(np.abs(W[j]).max())
            ims = Group(*[to_img(div_arr(W[j, i], m), 0.36) for i in range(4)]).arrange(RIGHT, buff=0.06)
            lbl = Text(f"F{j + 1}", font_size=16, color=GREY_B)
            rows.append(Group(lbl, ims).arrange(RIGHT, buff=0.15))
        fgrp = Group(*rows).arrange(DOWN, buff=0.08, aligned_edge=LEFT).move_to(LEFT * 4.9 + DOWN * 0.3)
        fhead = Text("8 filters x 4 input channels", font_size=20, color=GREY_B).next_to(fgrp, UP, buff=0.2)

        maps = Group(*[map_item(T["r2"][j], "gray", 1.3, f"F{j + 1}") for j in range(8)])
        maps.arrange_in_grid(2, 4, buff=0.35).move_to(RIGHT * 3.0 + DOWN * 0.2)
        arrow = Arrow(LEFT * 1.0, RIGHT * 1.0, buff=0).move_to(LEFT * 0.4 + DOWN * 0.2)

        self.play(FadeIn(fgrp), FadeIn(fhead))
        self.say("Layer 2 filters look at ALL 4 pooled maps at once and learn combinations")
        self.wait(2)
        self.play(FadeIn(arrow), FadeIn(maps), run_time=1.2)
        self.say("After conv + ReLU: 8 feature maps (11x11) of higher-level patterns")
        self.wait(2.5)

        pmaps = Group(*[map_item(T["p2"][j], "gray", 1.3, f"F{j + 1}") for j in range(8)])
        pmaps.arrange_in_grid(2, 4, buff=0.35).move_to(maps)
        self.play(FadeOut(maps), FadeIn(pmaps), run_time=1.2)
        self.say("Pool again -> 8 tiny maps of 5x5: a compact summary of the digit")
        self.wait(3)
        self.clear_stage()

    # ---------- 6. classifier ----------
    def sec_classifier(self):
        D, T = self.D, self.T
        self.set_title("6 | Flatten, classify, predict")

        pm = Group(*[map_item(T["p2"][j], "gray", 0.85) for j in range(8)])
        pm.arrange_in_grid(2, 4, buff=0.15).move_to(LEFT * 4.9 + DOWN * 0.2)
        pm_lab = Text("8 x 5 x 5", font_size=20, color=GREY_B).next_to(pm, UP, buff=0.15)
        self.play(FadeIn(pm), FadeIn(pm_lab))

        flat = T["flat"]
        strip = to_img(gray_arr(flat[:, None], 0, max(float(flat.max()), 1e-6)), 4.4)
        strip.stretch_to_fit_width(0.3)
        strip.move_to(LEFT * 1.6 + DOWN * 0.2)
        strip_fr = SurroundingRectangle(strip, buff=0, color=GREY_B, stroke_width=1.5)
        strip_lab = Text("200 numbers", font_size=20, color=GREY_B).next_to(strip, UP, buff=0.15)
        a1 = Arrow(LEFT * 2.9, LEFT * 2.0, buff=0).shift(DOWN * 0.2)
        self.play(FadeIn(a1), FadeIn(strip), Create(strip_fr), FadeIn(strip_lab))
        self.say("Flatten: unroll all feature maps into one long vector of 200 numbers")
        self.wait(2)

        fc = VGroup(RoundedRectangle(width=1.5, height=2.0, corner_radius=0.15, color=ORANGE),
                    Text("Linear\n200 -> 10", font_size=22)).move_to(RIGHT * 0.4 + DOWN * 0.2)
        a2 = Arrow(LEFT * 1.35, LEFT * 0.45, buff=0).shift(DOWN * 0.2)
        a3 = Arrow(RIGHT * 1.25, RIGHT * 2.05, buff=0).shift(DOWN * 0.2)
        self.play(FadeIn(a2), FadeIn(fc))
        self.say("Fully connected layer: weighs the evidence for each of the 10 digits")
        self.wait(1.5)

        # probability bars
        probs = T["probs"]
        pred = int(probs.argmax())
        bars, labels, pcts = [], [], []
        for i in range(10):
            y = 2.0 - i * 0.42
            w = max(float(probs[i]) * 3.0, 0.02)
            bar = Rectangle(width=w, height=0.28, stroke_width=0)
            bar.set_fill(GREEN if i == pred else BLUE_D, 1)
            bar.move_to([2.75 + w / 2, y, 0])
            lab = Text(str(i), font_size=22, color=GREEN if i == pred else WHITE).move_to([2.4, y, 0])
            pct = Text(f"{probs[i] * 100:.1f}%", font_size=18, color=GREY_A).next_to(bar, RIGHT, buff=0.1)
            bars.append(bar); labels.append(lab); pcts.append(pct)
        self.play(FadeIn(a3), *[FadeIn(l) for l in labels])
        self.play(*[GrowFromEdge(b, LEFT) for b in bars], *[FadeIn(p) for p in pcts], run_time=1.5)
        self.say("Softmax turns the 10 scores into probabilities that add up to 100%")
        self.wait(1.5)

        thumb = to_img(gray_arr(D["img"], 0, 1), 1.1).move_to(LEFT * 3.2 + DOWN * 2.6)
        thumb_fr = SurroundingRectangle(thumb, buff=0, color=WHITE, stroke_width=2)
        result = Text(f"Prediction: {pred}   ({probs[pred] * 100:.1f}% confident)   |   true label: {D['label']}",
                      font_size=30, color=GREEN)
        result.next_to(thumb_fr, RIGHT, buff=0.4)
        self.play(FadeIn(thumb), Create(thumb_fr), Write(result))
        self.play(Indicate(bars[pred], color=YELLOW, scale_factor=1.15))
        self.wait(3)
        self.clear_stage()

    # ---------- 7. summary ----------
    def sec_summary(self):
        self.set_title("The whole pipeline")
        names = ["Image\n28x28", "Conv\n4 filters", "ReLU", "Pool\n2x2", "Conv\n8 filters",
                 "ReLU", "Pool\n2x2", "Flatten\n200", "FC\n10", "Softmax"]
        colors = [WHITE, BLUE, YELLOW, GREEN, BLUE, YELLOW, GREEN, GREY_B, ORANGE, RED]
        blocks = VGroup()
        for n, c in zip(names, colors):
            box = RoundedRectangle(width=1.15, height=1.2, corner_radius=0.12, color=c, stroke_width=3)
            txt = Text(n, font_size=17).move_to(box)
            blocks.add(VGroup(box, txt))
        blocks.arrange(RIGHT, buff=0.18).scale_to_fit_width(13.2).move_to(UP * 0.3)
        arrows = VGroup(*[Arrow(blocks[i].get_right(), blocks[i + 1].get_left(), buff=0.02,
                                stroke_width=2, max_tip_length_to_length_ratio=0.5)
                          for i in range(len(blocks) - 1)])
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in blocks], lag_ratio=0.2),
                  FadeIn(arrows), run_time=3)
        self.say("Pixels -> edges -> parts of digits -> the digit itself")
        for b in blocks:
            self.play(Indicate(b, scale_factor=1.12), run_time=0.25)
        end = Text("Every filter and weight was learned from data.", font_size=34, color=YELLOW)
        end.move_to(DOWN * 1.6)
        self.play(Write(end))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])
