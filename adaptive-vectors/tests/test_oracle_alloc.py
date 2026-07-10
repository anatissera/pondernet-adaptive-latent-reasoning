def oracle_n_per_step(depths, m_max):
    return [min(d, m_max) for d in depths]

def test_oracle_clamps_to_m():
    assert oracle_n_per_step([1, 2, 3, 4], 4) == [1, 2, 3, 4]
    assert oracle_n_per_step([1, 2, 3, 4], 2) == [1, 2, 2, 2]

if __name__ == "__main__":
    test_oracle_clamps_to_m(); print("ALL PASS")
