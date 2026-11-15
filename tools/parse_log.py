p="tools/verify_curriculum.py"
import re
s=open(r"tools\retrain\curriculum_check.log",encoding="utf-8",errors="ignore").read()
pairs=re.findall(r"ep_rew_mean\s*\|\s*([-\d.e+]+).*?total_timesteps\s*\|\s*(\d+)", s, re.S)
if not pairs:
    pairs=re.findall(r"total_timesteps\s*\|\s*(\d+)|ep_rew_mean\s*\|\s*([-\d.e+]+)", s)
print("log lines:", len(pairs))
steps=[int(b) for a,b in pairs if b.isdigit()]
rews=[float(a) for a,b in pairs if b.isdigit()]
for i in range(0, len(steps), max(1,len(steps)//10)):
    print(steps[i], rews[i])
