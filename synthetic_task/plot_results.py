import os, re, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

batch_size = 1
BASE_DIR = f"../synthetic_results_{batch_size}_not_learnable_1"
METHODS = [
    "qpth",
    "ffoqp_eq_schur",
    "ffocp_eq"
]

def load_results(base_dir=BASE_DIR, methods=METHODS):
    dfs = []
    for m in methods:
        pattern = os.path.join(base_dir, m, "*.csv")
        for fp in sorted(glob.glob(pattern)):
            df = pd.read_csv(fp)
            df["method"] = m

            fname = os.path.basename(fp)
            def grab(pat, cast=float):
                mo = re.search(pat, fname)
                return cast(mo.group(1)) if mo else np.nan

            df["seed"] = grab(r"_seed(\d+)", int)
            df["ydim"] = grab(r"ydim(\d+)", int)
            df["lr"]   = grab(r"lr([0-9eE\.\-]+)", float)
            # df["eps"]  = grab(r"eps([0-9eE\.\-]+)", float)

            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSVs found under {base_dir}.")
    return pd.concat(dfs, ignore_index=True, sort=False)

df = load_results()
df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

# for c in ["epoch","test_ts_loss","test_df_loss","forward_time","backward_time"]:
#     if c not in df.columns:
#         df[c] = np.nan

def plot_time(df):
    
    print("Columns:", df.columns.tolist())
    print(df.head())
    # === 2. Compute averages over epochs and seeds ===
    df['total_time'] = df['forward_time'] + df['backward_time']
    grouped = (
        df.groupby(['method', 'ydim'], as_index=False)
        .agg({'forward_time': 'mean', 'backward_time': 'mean', 'total_time': 'mean'})
    )

    # === 3. Plot ===
    metrics = ['forward_time', 'backward_time', 'total_time']
    titles = ['Forward Time', 'Backward Time', 'Total Time']

    plt.figure(figsize=(8, 6))

    colors = {
        'qpth': 'tab:blue',
        'ffoqp_eq_schur': 'tab:orange',
        'ffocp_eq': 'tab:green'
    }

    for i, metric in enumerate(metrics, start=1):
        plt.subplot(3, 1, i)
        for method, color in colors.items():
            subset = grouped[grouped['method'] == method].sort_values('ydim')
            plt.plot(subset['ydim'], subset[metric], marker='o', label=method, color=color)
        plt.title(titles[i-1])
        plt.xlabel("ydim")
        plt.ylabel("Average Time")
        plt.grid(True)
        if i == 1:
            plt.legend()

    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/average_times.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    
plot_time(df)