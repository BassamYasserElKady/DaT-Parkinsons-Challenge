from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
source = root / "results" / "combined_histogram_clusters_10" / "best_10_cluster_assignments.csv"
out = root / "results" / "combined_histogram_clusters_10" / "cluster_uid_lists"
out.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(source).sort_values(["gmm_cluster", "uid"])
for cluster, group in data.groupby("gmm_cluster"):
    (out / f"cluster_{int(cluster):02d}_uids.txt").write_text(
        "\n".join(group["uid"].astype(str)) + "\n", encoding="utf-8"
    )
print(f"wrote {data.gmm_cluster.nunique()} cluster UID lists to {out}")
