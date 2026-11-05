# Kidney protocol v0.6 — rationale (DRAFT, Agent V, 2026-10-15; not a registration)

**Why two penalties.** Up to v0.5 the comparison model and the hypothesis block shared one ridge penalty, chosen by
patient-grouped inner CV for the whole design. With a 378-column comparison model and 24 patients, that one penalty is
set by what the wide comparison block needs, and the same shrinkage is then applied to the few columns of the hypothesis.
On Agent Q's benchmark worlds this shrank a true 2 % product to zero held-out gain (v05 median 0.000, 0 of 3 worlds
passing) although the same hypothesis scored 0.013 without the comparison model. Relation v06 fits one ridge over
[comparison | hypothesis block] with a separate penalty for each block, the pair chosen jointly by the same grouped inner
CV (6 × 6 grid, each grid ending in the zero model), and the competitor is the same two-penalty fit with the simpler
block. This removed the detection floor (2 % product 0.014, 2 of 3 worlds; held-out correlation of the gain's prediction
difference with the planted signal 0.83–0.99 instead of 0–0.56), kept the false-pass rate under a permuted hypothesis
block at 1.9 %, and gives the same answer as v05 where the comparison model is informative (the two v0.4 objects re-scored
to ~1e-4). Both penalties are recorded on every receipt so the reader can see when either block went to the zero model.

**Why the ladder is inside the run.** v0.4 and v0.5 put the whole comparison model in front of every hypothesis, which
makes "novel" mean "whatever survives after 378 variables have explained the target" and hides simple biology that
composition or known programmes would also explain. Agent D's post hoc ladder showed what this hides on the v0.4 pool
(25 simple associations and 18 partly absorbed against 1 irreducible). If the ladder is computed after the run, the
levels, nulls and pass rule are chosen after the outcome is known, and a hypothesis's tier can be argued into place.
Running it inside the registered run fixes all of that in advance: every frozen hypothesis is validated at N, B1, B2 and
B3 with its own held-out CI, held-out R², penalties, linear part and the registered nulls for that level, the levels are
built by the same function the ladder script uses (identical by construction and checked by a test), the tier follows a
written rule over the level passes, and no hypothesis is dropped — each is reported with its tier, its absorbing step and
the overlap/leak flags. The cost moves to validation (four levels × nulls per hypothesis), so the protocol fixes a budget
of frozen hypotheses per target and every receipt records the seconds spent at each level.
