"""Figures for RESULTS.md, generated from the run records.

Same rule as the report text: **nothing here is drawn by hand**, and every
mark traces to a file in `runs/`. A chart that has drifted from its data is
worse than no chart, because a picture is believed faster than a table.

Three figures, and only three. Each answers a question the writeup actually
asks. A feature that does not appear in the written summary should not be
built, and that applies to plots more than to most things —
they are the easiest thing in a repository to over-produce.

  ladder   cost-to-solve. Does the task get solved with less budget? The
           whole point of the iteration ladder, and the figure that decides
           whether the tasks discriminate at all.
  runs     every single run as one dot. Not a mean. Observed progress is
           bimodal — clusters at 0%, ~56%, ~93%, 100% — so a mean names a
           value no run took. This figure is `stats.describe()` drawn.
  rates    cheat and solve rates with Wilson intervals, forest-plot style.
           A rate without its interval is the claim this project exists to
           argue against.

Design notes, so the choices are not mistaken for defaults:

  * Colors are the first three slots of a validated categorical palette,
    checked with a CVD validator rather than chosen by eye (all-pairs
    Delta E 9.2 light / 9.4 dark against an >= 8 target). Three is the
    all-pairs ceiling for that palette; a fourth series folds into "other"
    rather than inventing a hue.
  * Every series is DIRECTLY LABELLED. On the light surface the aqua slot
    sits at 2.74:1 contrast, below the 3:1 bar, and the documented relief
    for that is visible labels — identity never rests on color alone.
  * Light and dark are separate renders from the same ramps, not an
    inverted image. GitHub picks between them with <picture>.
  * No dual axes, ever. Two measures of different scale get two figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentcheck.stats import wilson

#: Validated categorical slots. Light and dark are the same hues stepped
#: for their own surface — not a flip.
#:
#: THREE for all-pairs forms (scatter, dot plots, multi-line), where every
#: series can end up beside every other: that is the ceiling at which this
#: palette still clears the separation gates (all-pairs CVD Delta E 9.2
#: light / 9.4 dark against an >= 8 target).
#:
#: FIVE for adjacent forms (stacked bars), where only neighbouring segments
#: touch and the weaker pairs never meet: worst adjacent CVD Delta E 9.1
#: light / 8.4 dark. Validated separately, not assumed from the three.
#:
#: Past five, fold the tail into a labelled "other" class. A generated
#: sixth hue is indistinguishable from an existing one under CVD and
#: breaks every check the validator makes.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181")

#: All-pairs ceiling — dot plots, scatter, multi-line.
MAX_SERIES = 3
#: Adjacent-only ceiling — stacked bars.
MAX_STACKED = 5


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]


LIGHT = Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781",
              "#e1e0d9", "#c3c2b7", SERIES_LIGHT)
DARK = Theme("dark", "#1a1a19", "#ffffff", "#c3c2b7", "#898781",
             "#2c2c2a", "#383835", SERIES_DARK)
THEMES = (LIGHT, DARK)


def _axes(theme: Theme, size: tuple[float, float]) -> tuple[Any, Any]:
    """A figure and axis wearing the theme, with recessive chrome.

    Imported lazily: matplotlib is a dev extra, and the harness must run —
    and its whole test suite must pass — without it installed. A benchmark
    that cannot produce a number because a plotting library is missing has
    the dependency the wrong way round.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=size)
    # The value labels live outside the axes on the right. Without the
    # reserved margin `bbox_inches="tight"` grows the canvas instead, which
    # changes the aspect ratio between one figure and the next.
    fig.subplots_adjust(right=0.72)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme.muted, labelsize=9, length=0)
    ax.grid(True, axis="y", color=theme.grid, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig: Any, out: Path, theme: Theme, stem: str) -> Path:
    """Write the figure so the same data always produces the same bytes.

    Matplotlib's SVG output is not deterministic by default: it stamps the
    save time into `<dc:date>`, and names every clip path from a random
    salt. Every `report.py` run therefore rewrote ~1,200 lines across the
    twelve figures with no change in any number — a diff that buries a
    real change to a figure, in a repository whose claim is that its
    outputs are traceable to its inputs. A fixed salt and no date make the
    SVG a function of the data.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}-{theme.name}.svg"
    with matplotlib.rc_context({"svg.hashsalt": f"agentcheck-{stem}"}):
        fig.savefig(path, format="svg", bbox_inches="tight",
                    facecolor=theme.surface, metadata={"Date": None})
    plt.close(fig)
    return path


# --- the cost-to-solve curve ------------------------------------------------

def ladder(series: dict[str, list[tuple[int, int, int]]], out: Path,
           stem: str = "ladder") -> list[Path]:
    """Solve rate against iteration budget, one line per task.

    `series` maps a task id to (iteration cap, solved, scoreable) triples.

    Whiskers are Wilson intervals, drawn because at n=8 they are most of
    what there is to see: 8/8 is [68%, 100%], and a line through the point
    estimates alone would imply a precision the sample does not have.

    Reading the shape is the point. Flat and high means the task is solved
    at every budget and measures nothing; a knee means the budget is what
    was binding, and its position is task difficulty in a unit a reader can
    compare across tasks.
    """
    if len(series) > MAX_SERIES:
        raise ValueError(
            f"{len(series)} series exceeds the {MAX_SERIES}-slot all-pairs "
            f"ceiling of the validated palette. Facet into small multiples "
            f"or fold the tail into 'other' — a generated hue is "
            f"indistinguishable from an existing one under CVD.")

    written = []
    for theme in THEMES:
        fig, ax = _axes(theme, (7.0, 4.2))
        for i, (label, points) in enumerate(sorted(series.items())):
            color = theme.series[i]
            points = sorted(points)
            xs = [p[0] for p in points]
            ys = [k / n for _, k, n in points]
            lows = [y - wilson(k, n).low for y, (_, k, n) in zip(ys, points,
                                                                strict=True)]
            highs = [wilson(k, n).high - y for y, (_, k, n) in zip(ys, points,
                                                                  strict=True)]
            ax.errorbar(xs, ys, yerr=[lows, highs], color=color, linewidth=2.0,
                        marker="o", markersize=8, capsize=4, elinewidth=1.5,
                        zorder=3, label=label,
                        markeredgecolor=theme.surface, markeredgewidth=2)
            # Direct label at the last point. The relief for the aqua slot's
            # sub-3:1 contrast on the light surface, and it means identity
            # survives a greyscale print.
            ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(10, 0), color=color, fontsize=9,
                        va="center", annotation_clip=False)

        ax.set_xscale("log")
        ax.set_xticks(sorted({p[0] for pts in series.values() for p in pts}))
        ax.get_xaxis().set_major_formatter(
            __import__("matplotlib").ticker.ScalarFormatter())
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("iteration budget", color=theme.secondary, fontsize=10)
        ax.set_ylabel("runs solved cleanly", color=theme.secondary,
                      fontsize=10)
        ax.set_title("Cost to solve, with 95% Wilson intervals",
                     color=theme.ink, fontsize=12, loc="left", pad=12)
        ax.margins(x=0.12)
        written.append(_save(fig, out, theme, stem))
    return written


# --- every run, as a dot ----------------------------------------------------

def runs(groups: dict[str, list[float]], out: Path,
         stem: str = "runs") -> list[Path]:
    """One dot per run. Deliberately not a bar of means.

    Observed credible progress is bimodal — 0%, ~56%, ~93%, 100% — so a
    mean names a value no run took, and a bar chart of means would hide the
    single most reportable property of this data, which is its spread. The
    median is marked; the dots are the result.
    """
    written = []
    for theme in THEMES:
        labels = list(groups)
        fig, ax = _axes(theme, (7.0, 0.42 * max(len(labels), 3) + 1.4))
        for row, label in enumerate(labels):
            values = sorted(groups[label])
            color = theme.series[row % MAX_SERIES]
            # A deterministic vertical offset per duplicate value, so two
            # runs that both scored 100% are two visible dots rather than
            # one. Jitter here would be random noise drawn on top of data.
            seen: dict[float, int] = {}
            for v in values:
                k = seen.get(v, 0)
                seen[v] = k + 1
                ax.plot(v, row + (k - (values.count(v) - 1) / 2) * 0.11,
                        marker="o", markersize=9, color=color, zorder=3,
                        markeredgecolor=theme.surface, markeredgewidth=2)
            if values:
                mid = len(values) // 2
                median = (values[mid] if len(values) % 2
                          else (values[mid - 1] + values[mid]) / 2)
                ax.plot([median, median], [row - 0.3, row + 0.3],
                        color=theme.ink, linewidth=2.0, zorder=4)
                # annotation_clip=False, or matplotlib silently drops
                # every one of these: they sit outside the axes by design,
                # and the default is to clip there. The first render lost
                # all of them and looked fine.
                ax.annotate(f"median {median:.0%}   n={len(values)}",
                            (1.02, row), xycoords=("axes fraction", "data"),
                            color=theme.secondary, fontsize=8, va="center",
                            annotation_clip=False)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, color=theme.secondary, fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", color=theme.grid, linewidth=1.0)
        ax.invert_yaxis()
        ax.set_xlabel("credible progress", color=theme.secondary, fontsize=10)
        ax.set_title("Every run, not the mean", color=theme.ink,
                     fontsize=12, loc="left", pad=12)
        written.append(_save(fig, out, theme, stem))
    return written


# --- rates with intervals ---------------------------------------------------

def rates(rows: Sequence[tuple[str, int, int]], out: Path,
          stem: str = "rates") -> list[Path]:
    """Forest plot: a rate is never drawn without its interval.

    `rows` are (label, successes, n). The bar-chart alternative would put
    1/8 and 100/800 at the same height with nothing to say they are not
    equally known, which is precisely the reading this project argues
    against.
    """
    written = []
    for theme in THEMES:
        fig, ax = _axes(theme, (7.0, 0.42 * max(len(rows), 3) + 1.4))
        for row, (_label, k, n) in enumerate(rows):
            ci = wilson(k, n)
            color = theme.series[0]
            ax.plot([ci.low, ci.high], [row, row], color=color,
                    linewidth=2.0, solid_capstyle="round", zorder=3)
            ax.plot(ci.point, row, marker="o", markersize=9, color=color,
                    markeredgecolor=theme.surface, markeredgewidth=2,
                    zorder=4)
            ax.annotate(f"{k}/{n} = {ci.point:.0%}  "
                        f"[{ci.low:.0%}, {ci.high:.0%}]",
                        (1.02, row), xycoords=("axes fraction", "data"),
                        color=theme.secondary, fontsize=8, va="center",
                        annotation_clip=False)

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], color=theme.secondary,
                           fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", color=theme.grid, linewidth=1.0)
        ax.invert_yaxis()
        ax.set_xlabel("detector-flag rate, 95% Wilson interval", color=theme.secondary,
                      fontsize=10)
        ax.set_title("Detector flags (precision and recall unaudited)",
                     color=theme.ink, fontsize=12, loc="left", pad=12)
        written.append(_save(fig, out, theme, stem))
    return written


def picture(stem: str, alt: str, directory: str = "docs/figures") -> str:
    """The <picture> block that shows the right render per GitHub theme.

    Markdown's plain image syntax cannot switch on the reader's theme, and
    a light-surface chart on a dark page is the most common way a good
    figure becomes unreadable.
    """
    return (f'<picture>\n'
            f'  <source media="(prefers-color-scheme: dark)" '
            f'srcset="{directory}/{stem}-dark.svg">\n'
            f'  <img alt="{alt}" src="{directory}/{stem}-light.svg">\n'
            f'</picture>')


# --- what actually happened, not just whether it worked ---------------------

#: The nine recorded outcomes, folded into five reportable classes.
#:
#: The folding is an analytical claim, not a convenience, so it is written
#: here rather than buried in a plotting call:
#:
#:   * `gave_up` and `stopped_without_solving` are ONE class. Both mean the
#:     agent stopped while budget remained — it decided it was finished.
#:     One said so through the give_up tool and the other simply returned
#:     no tool calls, but the behaviour being measured is the same
#:     decision. `agent.py` keeps them apart at record time precisely so
#:     this figure can choose to join them; joining them at record time
#:     would have thrown the distinction away permanently.
#:
#:   * The three caps are ONE class. Which limit bit first is a fact about
#:     the harness configuration, not about the agent.
#:
#:   * `solved_with_cheats` stays ALONE, never merged into `solved`. It is
#:     the headline finding of the whole project and folding it into the
#:     success class is the exact mistake the project exists to name.
#:
#:   * `no_verdict` and `harness_error` are ONE class, and it is a class of
#:     its own rather than a failure: a measurement that did not happen is
#:     neither a success nor a failure, and counting it as either
#:     fabricates a result.
#:
#:   * `broke_suite` shares that class in THIS FIGURE and nowhere else.
#:     The agent editing the repository until pytest can no longer collect
#:     it is a result, not a missing measurement, and RESULTS.md separates
#:     the two everywhere it counts them. It is not a sixth segment here
#:     because the palette validates to five under CVD and a generated
#:     sixth hue would be indistinguishable from an existing one — a
#:     grouping made for a stated reason, not an oversight. If this figure
#:     ever needs the split, facet it; do not add a colour.
#:
#: Order is worst-to-best left-to-right in the stack, so the eye reads the
#: bar as a quality gradient.
OUTCOME_CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("solved cleanly", ("solved",)),
    ("green but flagged", ("solved_with_cheats",)),
    ("stopped short", ("gave_up", "stopped_without_solving")),
    ("hit a cap", ("iteration_limit", "cost_limit", "time_limit")),
    ("no verdict or broke the suite",
     ("no_verdict", "harness_error", "broke_suite")),
)

_CLASS_OF = {outcome: name
             for name, outcomes in OUTCOME_CLASSES
             for outcome in outcomes}


def classify(outcome: str) -> str:
    """The reportable class of a recorded outcome.

    Raises on an unknown outcome rather than bucketing it. A new outcome
    silently swept into "no verdict" would be a behaviour the project added
    on purpose and then hid from itself.
    """
    try:
        return _CLASS_OF[outcome]
    except KeyError:
        raise KeyError(
            f"outcome {outcome!r} has no reportable class. Add it to "
            f"OUTCOME_CLASSES deliberately — defaulting it into an existing "
            f"class would hide a behaviour someone added on purpose."
        ) from None


def outcomes(rungs: Sequence[tuple[str, list[str]]], out: Path,
             stem: str = "outcomes") -> list[Path]:
    """Outcome composition per iteration budget, as a stacked bar.

    `rungs` are (label, list of recorded outcomes) in the order to plot.

    This is the figure the ladder is really for. The solve-rate curve
    collapses every way a run can end into one number, and the ways differ:
    squeezing the budget from 100 iterations to 5 turning "solved" into
    "hit a cap" means the budget was binding, which is a statement about
    the task. The same squeeze turning "solved" into "stopped short" means
    the agent gave up early with budget in hand, which is a statement about
    the model. A line cannot tell those apart; a stack can.

    Proportions, not counts, so rungs with different n stay comparable —
    with n printed on each bar, because a proportion without its
    denominator is the oldest way to overstate a result.
    """
    written = []
    for theme in THEMES:
        labels = [r[0] for r in rungs]
        fig, ax = _axes(theme, (7.4, 0.52 * max(len(rungs), 3) + 1.6))
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", color=theme.grid, linewidth=1.0)

        for row, (_label, recorded) in enumerate(rungs):
            n = len(recorded)
            if not n:
                continue
            counts = {name: sum(1 for o in recorded if classify(o) == name)
                      for name, _ in OUTCOME_CLASSES}
            left = 0.0
            for i, (name, _) in enumerate(OUTCOME_CLASSES):
                share = counts[name] / n
                if not share:
                    continue
                # A 2px surface-coloured gap between segments, so adjacent
                # fills never read as one block — the separation that lets
                # the palette clear its gates on the adjacent pairlist.
                ax.barh(row, share, left=left, height=0.62,
                        color=theme.series[i], zorder=3,
                        edgecolor=theme.surface, linewidth=2)
                # Direct value labels: mandatory past three series, and the
                # documented relief for the three light-mode slots that sit
                # under 3:1 contrast. Only where the segment can hold text.
                if share >= 0.12:
                    ax.annotate(f"{counts[name]}", (left + share / 2, row),
                                ha="center", va="center", fontsize=9,
                                color=theme.surface, fontweight="bold",
                                zorder=4)
                left += share
            ax.annotate(f"n={n}", (1.02, row),
                        xycoords=("axes fraction", "data"),
                        color=theme.secondary, fontsize=8, va="center",
                        annotation_clip=False)

        # A legend is always present past one series; the value labels are
        # the second channel, so identity never rests on colour alone.
        import matplotlib.patches as mpatches
        present = {classify(o) for _, rec in rungs for o in rec}
        handles = [mpatches.Patch(color=theme.series[i], label=name)
                   for i, (name, _) in enumerate(OUTCOME_CLASSES)
                   if name in present]
        legend = ax.legend(handles=handles, loc="upper center",
                           bbox_to_anchor=(0.5, -0.22), ncol=3,
                           frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(theme.secondary)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, color=theme.secondary, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.invert_yaxis()
        ax.set_xlabel("share of runs", color=theme.secondary, fontsize=10)
        ax.set_title("How runs ended, by iteration budget",
                     color=theme.ink, fontsize=12, loc="left", pad=12)
        written.append(_save(fig, out, theme, stem))
    return written


# --- was the cap ever binding? ----------------------------------------------

def budget(series: dict[str, list[tuple[int, int]]], out: Path,
           stem: str = "budget") -> list[Path]:
    """Iterations used against the cap allowed, one dot per run.

    `series` maps a task id to (cap, iterations used) pairs.

    A validity check on the ladder itself, and it is worth having before
    the results are read rather than after. The archive's runs stop at
    39-91 iterations under a cap of 150, so that cap was never binding and
    every number produced under it describes the agent's own stopping
    behaviour, not its budget.

    The diagonal is `used == cap`. A dot ON it is a run the budget ended; a
    dot BELOW it is a run that stopped on its own. If the top rung's dots
    all sit below the line, that rung is genuinely unconstrained and works
    as the control the ladder needs. If they sit on it, the ladder has no
    control rung and the top needs raising — which is much cheaper to learn
    from this figure than from a conclusion drawn on a bad design.
    """
    if len(series) > MAX_SERIES:
        raise ValueError(
            f"{len(series)} series exceeds the {MAX_SERIES}-slot all-pairs "
            f"ceiling: every dot can land beside every other here, so the "
            f"adjacent-only slots do not apply. Facet instead.")

    written = []
    for theme in THEMES:
        fig, ax = _axes(theme, (6.6, 4.4))
        caps = sorted({c for pts in series.values() for c, _ in pts})
        if not caps:
            raise ValueError("no runs to plot")

        # The reference line first, so data sits above it. Drawn across the
        # rungs themselves rather than a padded range, so its endpoints are
        # real positions on the axis.
        ax.plot([min(caps), max(caps)], [min(caps), max(caps)],
                color=theme.axis, linewidth=1.5, linestyle="--", zorder=2)
        ax.annotate("used the whole budget", (max(caps), max(caps)),
                    textcoords="offset points", xytext=(8, -4),
                    color=theme.muted, fontsize=8, annotation_clip=False,
                    va="center")

        for i, (label, points) in enumerate(sorted(series.items())):
            color = theme.series[i]
            xs = [c for c, _ in points]
            ys = [u for _, u in points]
            ax.plot(xs, ys, linestyle="none", marker="o", markersize=8,
                    color=color, markeredgecolor=theme.surface,
                    markeredgewidth=2, zorder=3, label=label)
            if points:
                last = max(points)
                ax.annotate(label, last, textcoords="offset points",
                            xytext=(10, 4), color=color, fontsize=9,
                            annotation_clip=False)

        import matplotlib.ticker as mticker
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Log axes label their MINOR ticks too, which at this data range
        # renders "1.6 x 10^2 1.7 x 10^2 1.8 x 10^2" overlapping into an
        # unreadable smear. Only the rungs are meaningful positions here,
        # so only the rungs get labels.
        ax.set_xticks(caps)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())

        # Round, human y ticks spanning what was observed. The default log
        # locator emits "6 x 10^1", which nobody reads as 60.
        lo = min(u for pts in series.values() for _, u in pts)
        hi = max(u for pts in series.values() for _, u in pts)
        ticks = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500)
                 if lo / 1.6 <= v <= hi * 1.6]
        if ticks:
            ax.set_yticks(ticks)
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_yaxis().set_minor_formatter(mticker.NullFormatter())

        # With a single rung the automatic range expands to a decade either
        # side and strands the data in the middle. Bound it to the rungs.
        ax.set_xlim(min(caps) * 0.75, max(caps) * 1.45)
        ax.grid(True, axis="both", color=theme.grid, linewidth=1.0)
        ax.set_xlabel("iteration cap", color=theme.secondary, fontsize=10)
        ax.set_ylabel("iterations actually used", color=theme.secondary,
                      fontsize=10)
        ax.set_title("Was the budget ever what stopped the run?",
                     color=theme.ink, fontsize=12, loc="left", pad=12)
        written.append(_save(fig, out, theme, stem))
    return written


# --- where the budget actually goes -----------------------------------------

def effort(rungs: Sequence[tuple[str, list[tuple[int, int]]]], out: Path,
           stem: str = "effort") -> list[Path]:
    """Reads against writes per iteration budget, as paired bars.

    `rungs` are (label, [(reads, writes), ...]) in the order to plot.

    The figure that explains the solve-rate curve instead of restating it.
    Reads saturate — 48.4 at a 40-iteration budget, 51.9 at 100, barely
    moving — while writes more than double over the same span. The agent
    spends its first fifty-odd iterations reading and only then begins to
    edit, so a budget below that never reaches the editing phase at all.

    That turns "it needs more iterations" into something falsifiable and
    more interesting: it needs to stop reading sooner. docs/findings.md §2
    records a prompt change moving writes from 0,0,8,0 to 8,8,11,7 with
    everything else held constant, which is the same claim from the other
    direction.

    Two series, so the palette's first two slots and a legend; grouped
    bars rather than stacked, because reads and writes are not parts of a
    whole and stacking them would invite reading their sum as a total
    effort that means nothing.
    """
    written = []
    for theme in THEMES:
        labels = [r[0] for r in rungs]
        fig, ax = _axes(theme, (7.2, 0.62 * max(len(rungs), 3) + 1.8))
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", color=theme.grid, linewidth=1.0)

        height = 0.34
        for row, (_label, pairs) in enumerate(rungs):
            if not pairs:
                continue
            n = len(pairs)
            means = (sum(p[0] for p in pairs) / n, sum(p[1] for p in pairs) / n)
            never_wrote = sum(1 for p in pairs if p[1] == 0)
            for i, (name, value) in enumerate(zip(("reads", "writes"), means,
                                                  strict=True)):
                offset = (i - 0.5) * (height + 0.04)
                ax.barh(row + offset, value, height=height,
                        color=theme.series[i], zorder=3,
                        edgecolor=theme.surface, linewidth=2,
                        label=name if row == 0 else None)
                ax.annotate(f"{value:.1f}", (value, row + offset),
                            textcoords="offset points", xytext=(6, 0),
                            va="center", fontsize=8, color=theme.secondary,
                            annotation_clip=False)
            # The count that carries the finding: how many runs at this
            # budget never edited anything at all.
            ax.annotate(f"{never_wrote}/{n} never wrote a file",
                        (1.02, row), xycoords=("axes fraction", "data"),
                        color=theme.secondary, fontsize=8, va="center",
                        annotation_clip=False)

        legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                           ncol=2, frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(theme.secondary)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, color=theme.secondary, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("mean tool calls per run", color=theme.secondary,
                      fontsize=10)
        ax.set_title("Where the budget goes: reading, not editing",
                     color=theme.ink, fontsize=12, loc="left", pad=12)
        written.append(_save(fig, out, theme, stem))
    return written
