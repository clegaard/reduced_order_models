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


# Resources
http://web.mit.edu/6.242/www/images/lec6_6242_2004.pdf
https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a


# Implement Implicit scheme

Define Diffusion operator K.
This would require a stencil like 1 -2 1 for the 1D case

# Implement Galerkin projection

