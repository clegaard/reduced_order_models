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

# temporal averaging

I would expect that U = [u0 u1 u2 ... un], where u_k are (N x M) snapshots of the system.
Wikipedia suggests that the signals are averaged for every point in space:
https://en.wikipedia.org/wiki/Proper_orthogonal_decomposition

I would expect the method to work like:
U(t, u0) = [z0, z1 z2 ... zm] * ϕ(t)

where z_i represents a 2-dimensional stencil that is modulated over time by ϕ(t) to reconstruct an component of the signal.


# Resources
http://web.mit.edu/6.242/www/images/lec6_6242_2004.pdf
https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a