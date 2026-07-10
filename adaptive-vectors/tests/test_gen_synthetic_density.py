import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import gen_synthetic_density as g

def _eval_block(block: str):
    # block looks like "<<EXPR=RESULT>>"
    inner = block[2:-2]
    expr, result = inner.rsplit("=", 1)
    return expr, float(result)

def test_arithmetic_correct():
    rng = random.Random(0)
    for _ in range(500):
        inst = g.gen_instance(4, [2, 3, 1, 4], rng)
        blocks = inst["cot"].split()
        assert len(blocks) == 4, blocks
        last = None
        for blk in blocks:
            expr, result = _eval_block(blk)
            assert abs(eval(expr) - result) < 1e-6, (expr, result)
            last = result
        assert float(inst["answer"]) == last
        assert all(v > 0 for _, v in [(_eval_block(b)) for b in blocks])
        assert last <= g.MAX_VAL, f"Result {last} exceeds MAX_VAL={g.MAX_VAL}"

def test_depths_recorded():
    rng = random.Random(1)
    inst = g.gen_instance(4, [1, 4, 2, 3], rng)
    assert inst["depths"] == [1, 4, 2, 3]
    # depth d step has exactly d operators
    for blk, d in zip(inst["cot"].split(), inst["depths"]):
        expr = blk[2:].rsplit("=", 1)[0]
        n_ops = sum(expr.count(op) for op in "+-*/")
        assert n_ops == d, (expr, d)

def test_question_has_composed_expr():
    rng = random.Random(2)
    inst = g.gen_instance(4, [2, 2, 2, 2], rng)
    assert inst["question"].startswith("Calculate:")
    # the composed question expression evaluates to the answer
    expr = inst["question"].split("Calculate:", 1)[1].strip()
    assert abs(eval(expr) - float(inst["answer"])) < 1e-6

def test_determinism():
    a = g.gen_instance(4, [1, 2, 3, 4], random.Random(7))
    b = g.gen_instance(4, [1, 2, 3, 4], random.Random(7))
    assert a == b

def test_plus_respects_max_val():
    """Regression: apply_op '+' branch must respect MAX_VAL bound."""
    for seed in range(200):
        rng = random.Random(seed)
        new_val = g.apply_op(995, rng)[1]
        assert new_val <= g.MAX_VAL, f"seed {seed}: 995 + op yielded {new_val} > {g.MAX_VAL}"

def test_level_means():
    import statistics
    for lvl in ["L0", "L1", "L2", "L3", "train"]:
        rng = random.Random(42)
        ds = [d for _ in range(4000) for d in g.sample_depths(lvl, 4, rng)]
        assert abs(statistics.mean(ds) - 2.5) < 0.1, (lvl, statistics.mean(ds))
    # variance is monotone increasing L0 < L1 < L2 < L3
    def var(lvl):
        rng = random.Random(1)
        ds = [d for _ in range(4000) for d in g.sample_depths(lvl, 4, rng)]
        return statistics.pvariance(ds)
    vs = [var(l) for l in ["L0", "L1", "L2", "L3"]]
    assert vs[0] < vs[1] < vs[2] < vs[3], vs

def test_build_split_unique_and_disjoint():
    train, qtrain = g.build_split("train", 500, 4, seed=0, exclude=set())
    test, qtest = g.build_split("L2", 200, 4, seed=1, exclude=qtrain)
    assert len({r["question"] for r in train}) == len(train)
    assert qtrain.isdisjoint(qtest)

def test_build_split_returns_exactly_n():
    rows, _ = g.build_split("L3", 300, 4, seed=5, exclude=set())
    assert len(rows) == 300

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("ALL PASS")
