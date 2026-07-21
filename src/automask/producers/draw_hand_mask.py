    shapes: list = []
    jf = os.path.join(MSK_DIR, f"hand_run{run:04d}.json")
    if os.path.exists(jf):
        with open(jf) as f:
            shapes = [(s["kind"], (tuple(s["geom"]) if s["kind"] in ("rect", "line")
                                   else [tuple(v) for v in s["geom"]])) for s in json.load(f)]
        print(f"[load] resumed {len(shapes)} shapes from {jf}")

    # in-progress state: rectangle first-corner, or accumulating polygon vertices
    state = {"mode": "rect", "corner": None, "poly": [], "temp": []}

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(disp, cmap="gray", origin="upper")
    ov = ax.imshow(_overlay(zero, shapes_to_mask(shapes)), origin="upper")
    ttl = ax.set_title("")
    ax.set_xlabel("col"); ax.set_ylabel("row")

    def clear_temp():
        for a in state["temp"]:
            a.remove()
        state["temp"] = []

    def refresh():
        hand = shapes_to_mask(shapes)
        ov.set_data(_overlay(zero, hand))
        tgt = zero | hand
        prog = (f"  {state['mode']}: 1st point set" if state["corner"] else
                (f"  poly: {len(state['poly'])} verts (enter to close)" if state["poly"] else ""))
        ttl.set_text(f"run {run}  [{state['mode']} mode]{prog}\n"
                     f"shapes={len(shapes)}  hand={100*hand.mean():.2f}%  "
                     f"target={100*tgt.mean():.2f}%   "
                     f"r/l/p=mode  enter=close-poly  esc=cancel  u=undo  c=clear  s=save  q=quit")
        fig.canvas.draw_idle()

    def cancel_progress():
        state["corner"] = None; state["poly"] = []; clear_temp()

    def set_mode(mode):
        state["mode"] = mode; cancel_progress(); refresh()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        tb = getattr(fig.canvas, "toolbar", None)          # ignore clicks while pan/zoom active
        if tb is not None and getattr(tb, "mode", ""):
            return
        r, c = event.ydata, event.xdata
        if state["mode"] in ("rect", "line"):        # both are two-click shapes
            if state["corner"] is None:
                state["corner"] = (r, c)
                state["temp"].append(ax.plot(c, r, "y+", ms=12, mew=2)[0])
            else:
                r0, c0 = state["corner"]
                if state["mode"] == "rect":
                    rr0, rr1 = sorted((int(round(r0)), int(round(r))))
                    cc0, cc1 = sorted((int(round(c0)), int(round(c))))
                    shapes.append(("rect", (rr0, rr1, cc0, cc1)))
                else:
                    shapes.append(("line", (int(round(r0)), int(round(c0)),
                                            int(round(r)), int(round(c)))))
                state["corner"] = None; clear_temp()
            refresh()
        else:  # polygon
            state["poly"].append((float(r), float(c)))
            xs = [p[1] for p in state["poly"]]; ys = [p[0] for p in state["poly"]]
            clear_temp()
            state["temp"].append(ax.plot(xs, ys, "y.-", ms=8, lw=1)[0])
            refresh()

    def save():
        write_target(run, shapes, fig=fig)

    def on_key(event):
        k = event.key