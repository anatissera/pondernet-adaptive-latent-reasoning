import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import analyze_synth_density as a

def test_interp_uniform_midpoint():
    cur = [(4.0, 20.0), (8.0, 40.0), (12.0, 50.0), (16.0, 54.0)]
    assert abs(a.interp_uniform(cur, 6.0) - 30.0) < 1e-6   # halfway between (4,20)-(8,40)
    assert abs(a.interp_uniform(cur, 8.0) - 40.0) < 1e-6
    assert abs(a.interp_uniform(cur, 2.0) - 20.0) < 1e-6   # clamp below
    assert abs(a.interp_uniform(cur, 20.0) - 54.0) < 1e-6  # clamp above

def test_pearson_perfect():
    assert abs(a.pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9
    assert abs(a.pearson([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9

def test_pearson_zero():
    # constant y → undefined, should return 0.0
    assert a.pearson([1, 2, 3], [5, 5, 5]) == 0.0

def test_level_variance_monotone():
    v0 = a.level_variance([[2, 3, 2, 3]] * 50)
    v3 = a.level_variance([[1, 4, 1, 4]] * 50)
    assert v3 > v0

if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f): f(); print(f"PASS {n}")
    print("ALL PASS")
