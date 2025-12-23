import pandas as pd
import numpy as np

# Creating the dataset
data = [
    ['Technical', 'Senior', 'excellent', 'good', 'urban', 'yes'],
    ['Technical', 'Junior', 'excellent', 'good', 'urban', 'yes'],
    ['Non-Technical', 'Junior', 'average', 'poor', 'rural', 'no'],
    ['Technical', 'Senior', 'average', 'good', 'rural', 'no'],
    ['Technical', 'Senior', 'excellent', 'good', 'rural', 'yes']
]

columns = ['Role', 'Experience', 'Performance', 'InternetQuality', 'WorkLocation', 'Output']

df = pd.DataFrame(data, columns=columns)


X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])


def is_consistent(hypothesis, example):
    return all(h == '?' or h == e for h, e in zip(hypothesis, example))


def more_general(h1, h2):
    return all(h1[i] == '?' or h1[i] == h2[i] for i in range(len(h1)))


def candidate_elimination(X, y):
    n = X.shape[1]

    # Initialize S and G
    S = X[y == 'yes'][0].copy()
    G = [['?' for _ in range(n)]]

    for i in range(len(X)):
        if y[i] == 'yes':
            # Generalize S
            for j in range(n):
                if S[j] != X[i][j]:
                    S[j] = '?'

            # Remove inconsistent hypotheses from G
            G = [g for g in G if is_consistent(g, X[i])]

        else: # Negative example
            new_G = []
            for g in G:
                if is_consistent(g, X[i]):
                    for j in range(n):
                        if S[j] != '?' and S[j] != X[i][j]:
                            new_h = g.copy()
                            new_h[j] = S[j]
                            if new_h not in new_G:
                                new_G.append(new_h)
                else:
                    new_G.append(g)

            # Keep only maximally general hypotheses
            G = [
                h for h in new_G
                if any(more_general(h, s) for s in [S])
            ]

            # Remove subsumed hypotheses
            G = [
                h for h in G
                if not any(
                    other != h and more_general(other, h)
                    for other in G
                )
            ]

        print(f"\nAfter instance {i+1}:")
        print("S =", S)
        print("G =", G)

    return S, G



final_S, final_G = candidate_elimination(X, y)

print("\nFinal Specific Hypothesis:", final_S)
print("Final General Hypothesis:", final_G)