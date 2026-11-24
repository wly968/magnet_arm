import numpy as np
d=np.load("scene_magnet/saves/eval_logs_p6/evaluations.npz")
ts=d["timesteps"]; res=d["results"]; lens=d["ep_lengths"]
print("eval 次数", len(ts))
for i in range(len(ts)):
    print("  %8d 步 | 平均回报 %8.1f | 长度 %6.1f | 成功(>=50) %2d/%d"
          % (ts[i], res[i].mean(), lens[i].mean(), int((res[i]>50).sum()), len(res[i])))
