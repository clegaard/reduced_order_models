# higher-order derivatives
why is this not the same?
``` python
# dudx, dudy = np.gradient(cur, Δx, Δy)
# dudxx, dudxy = np.gradient(dudx)
# dudyy, dudyx = np.gradient(dudy)
dudx = np.gradient(cur,Δx,axis=0)
dudxx = np.gradient(dudx, axis=0)
dudy = np.gradient(cur,Δy,axis=1)
dudyy = np.gradient(dudy, axis=1)
```