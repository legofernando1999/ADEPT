# PDE for dissolved asphaltene concentration 

$$\frac{\partial C_f}{\partial \tau} + \frac{\partial C_f}{\partial Z} + Da_p C_f = Da_p C_{eq} $$

Now let's use the finite difference method to solve this PDE

**Forward-difference method, jth step in time:** 
$$\frac{w_{i,j+1} - w_{i,j}}{\Delta \tau _{i,j}} + \left( \frac{w_{i,j} - w_{i-1,j}}{ \Delta Z} \right) + Da_p^{i,j} w_{i,j} = Da_p^{i,j} C_{eq}^{i,j}$$

**Backward-difference method, (j+1)st step in time:** 
$$\frac{w_{i,j+1} - w_{i, j}}{\Delta \tau _{i,j+1}} + \left( \frac{w_{i,j+1} - w_{i-1,j+1}}{ \Delta Z} \right) + Da_p^{i,j+1} w_{i,j+1} = Da_p^{i,j+1} C_{eq}^{i,j+1}$$

**Crank-Nicolson (averaged) method:**

Since we only have information from the jth step in time, we will assume that $\Delta \tau _{i,j+1} \approx \Delta \tau _{i,j}$, $Da_p^{i,j+1}\approx Da_p^{i,j}$ and $C_{eq}^{i,j+1} \approx C_{eq}^{i,j}$

$$\frac{w_{i, j+1} - w_{i, j}}{\Delta \tau _{i,j}} = -\frac{1}{2} \left( \frac{w_{i,j+1} - w_{i-1,j+1}}{ \Delta Z} + \frac{w_{i,j} - w_{i-1,j}}{ \Delta Z} \right) - \frac{Da_p^{i,j}}{2} \left( w_{i,j+1} + w_{i,j} \right) + Da_p^{i,j} C_{eq}^{i,j}$$

$$\Rightarrow w_{i,j+1} - w_{i, j} = -\alpha^{i,j}\left( w_{i,j+1} - w_{i-1,j+1} + w_{i,j} - w_{i-1,j} \right) - \beta^{i,j}\left( w_{i,j+1} + w_{i,j} \right) + \gamma^{i,j}$$

Where

$$\alpha^{i,j} = \frac{\Delta \tau _{i,j}}{2 \Delta Z}, \quad \beta^{i,j} = \frac{\Delta \tau _{i,j} Da_p^{i,j}}{2}, \quad \gamma^{i,j} = \Delta \tau _{i,j} Da_p^{i,j} C_{eq}^{i,j}$$

$$\Rightarrow w_{i,j+1} + \alpha^{i,j}\left( w_{i,j+1} - w_{i-1,j+1} \right) + \beta^{i,j} w_{i,j+1} = w_{i, j} + \alpha^{i,j} \left( w_{i-1,j} - w_{i,j}\right) - \beta^{i,j} w_{i,j}  + \gamma^{i,j}$$

$$\Rightarrow \left( 1 + \alpha^{i,j} + \beta^{i,j} \right) w_{i,j+1} - \alpha^{i,j} w_{i-1,j+1} = \left( 1 - \alpha^{i,j} - \beta^{i,j} \right) w_{i, j} + \alpha^{i,j} w_{i-1,j} + \gamma^{i,j}$$

We select an integer N > 0 and divide the interval $[0, 1]$ into $N-1$ equal subintervals whose endpoints are the mesh points $z_i = ih$, for $i = 0, 1, \ldots , N-1$, where $h = 1/(N-1)$

We solve the PDE at the mesh points, $z_i$, for $i = 1, 2, \ldots , N-1$

The resulting system of equations is expressed in the tridiagonal matrix form $A\textbf{w}^{(j+1)} = B\textbf{w}^{(j)} + \textbf{b}$ 

Where

$$\textbf{w}^{(j)} = \begin{pmatrix}
        w_{1,j}\\
        w_{2,j}\\
        \vdots\\
        w_{N-2,j}\\
        w_{N-1,j}
    \end{pmatrix}, \quad
\textbf{b} = \begin{pmatrix}
        \alpha^{1,j} \left( w_{0,j+1} + w_{0,j} \right) + \gamma^{1,j}\\
        \gamma^{2,j}\\
        \vdots\\
        \gamma^{N-2,j}\\
        \gamma^{N-1,j}
    \end{pmatrix}$$



# PDE for primary particle concentration
$$\frac{\partial C}{\partial \tau} = \frac{1}{Pe} \frac{\partial^2 C}{\partial Z^2} - \frac{\partial C}{\partial Z} + Da_p r_p - Da_{ag} C^2 - Da_d C$$

where:
$$r_p = \begin{cases}
        C_f - C_{eq}, &  C_f \geq C_{eq}\\
        -k_{diss} C, & C_f < C_{eq}
    \end{cases}$$

Before solving this PDE using a finite difference method we must rewrite it in a more suitable form with the help of the Heaviside function.

$$\frac{\partial C}{\partial \tau} = \frac{1}{Pe} \frac{\partial^2 C}{\partial Z^2} - \frac{\partial C}{\partial Z} + Da_p \left[ C_{df} \cdot H_1\left( C_{df} \right) - k_{diss} C \cdot H_0\left( -C_{df} \right)\right]\\ - Da_{ag} C^2 - Da_d C$$

Where
$$C_{df} = Cf - C_{eq}$$

and

$$H_a(x) = \begin{cases}
        0, & x < 0\\
        a, & x = 0\\
        1, & x > 0
\end{cases}$$

**Forward-difference method, jth step in time:** 
$$\frac{w_{i, j+1} - w_{i, j}}{\Delta \tau _{i,j}} = \frac{1}{Pe^{i,j}} \left( \frac{w_{i+1,j} - 2 w_{i,j} + w_{i-1,j}}{(\Delta Z)^2} \right) - \left( \frac{w_{i,j} - w_{i-1,j}}{ \Delta Z} \right)\\ + Da_p^{i,j}  \left[ F_{i,j} \cdot H_1\left( F_{i,j} \right) - k_{diss} w_{i,j} H_0\left( -F_{i,j} \right)\right] - Da_{ag}^{i,j} w_{i,j}^2 - Da_d^{i,j} w_{i,j}$$

**Backward-difference method, (j+1)st step in time:** 
$$\frac{w_{i, j+1} - w_{i, j}}{\Delta \tau _{i,j+1}} = \frac{1}{Pe^{i,j+1}} \left( \frac{w_{i+1,j+1} - 2 w_{i,j+1} + w_{i-1,j+1}}{(\Delta Z)^2} \right) - \left( \frac{w_{i,j+1} - w_{i-1,j+1}}{ \Delta Z} \right)\\ + Da_p^{i,j+1}  \left[ F_{i,j+1} \cdot H_1\left( F_{i,j+1} \right) - k_{diss} w_{i,j+1} H_0\left( -F_{i,j+1} \right) \right] - Da_{ag}^{i,j+1} w_{i,j+1}^2 - Da_d^{i,j+1} w_{i,j+1}$$

**Crank-Nicolson (averaged) method:**

Since we only have information from the jth step in time, we will assume that $\Delta \tau _{i,j+1}\approx \Delta \tau _{i,j}$, $Pe^{i,j+1} \approx Pe^{i,j}$, $Da_p^{i,j+1} \approx Da_p^{i,j}$, $Da_{ag}^{i,j+1} \approx Da_{ag}^{i,j}$, $Da_d^{i,j+1} \approx Da_d^{i,j}$ and $C_{eq}^{i,j+1} \approx C_{eq}^{i,j}$ (so $F_{i,j}$ and $F_{i,j+1}$ are defined with $C_{eq}^{i,j}$)

$$\frac{w_{i, j+1} - w_{i, j}}{\Delta \tau _{i,j}} = \frac{1}{2 Pe^{i,j}} \left( \frac{w_{i+1,j+1} - 2 w_{i,j+1} + w_{i-1,j+1}}{(\Delta Z)^2} + \frac{w_{i+1,j} - 2 w_{i,j} + w_{i-1,j}}{(\Delta Z)^2} \right)\\ - \frac{1}{2}\left( \frac{w_{i,j+1} - w_{i-1,j+1}}{ \Delta Z}+ \frac{w_{i,j} - w_{i-1,j}}{ \Delta Z} \right)\\ + \frac{Da_p^{i,j}}{2} \left[ F_{i,j+1} \cdot H_1\left( F_{i,j+1} \right) - k_{diss} w_{i,j+1} H_0\left( -F_{i,j+1} \right) + F_{i,j} \cdot H_1\left( F_{i,j} \right) - k_{diss} w_{i,j} H_0\left( -F_{i,j} \right) \right]\\ - \frac{Da_{ag}^{i,j}}{2} \left( w_{i,j+1}^2 + w_{i,j}^2 \right) - \frac{Da_d^{i,j}}{2} \left( w_{i,j+1} + w_{i,j} \right)$$

$$\Rightarrow w_{i, j+1} - w_{i, j} = \alpha_{i,j} \left( w_{i+1,j+1} - 2 w_{i,j+1} + w_{i-1,j+1} + w_{i+1,j} - 2 w_{i,j} + w_{i-1,j} \right)\\ - \beta_{i,j} \left( w_{i,j+1} - w_{i-1,j+1} + w_{i,j} - w_{i-1,j} \right)\\ + \gamma_{i,j} \left[ F_{i,j+1} \cdot H_1\left( F_{i,j+1} \right) - k_{diss} w_{i,j+1} H_0\left( -F_{i,j+1} \right) + F_{i,j} \cdot H_1\left( F_{i,j} \right) - k_{diss} w_{i,j} H_0\left( -F_{i,j} \right) \right]\\ - \delta_{i,j} \left( w_{i,j+1}^2 + w_{i,j}^2 \right) - \epsilon_{i,j} \left( w_{i,j+1} + w_{i,j} \right)$$

Where:

$$\alpha_{i,j} = \frac{\Delta \tau _{i,j}}{2 Pe^{i,j} (\Delta Z)^2}, \quad \beta_{i,j} = \frac{\Delta \tau_{i,j}}{2 \Delta Z}, \quad \gamma_{i,j} = \frac{\Delta \tau_{i,j} Da_p^{i,j}}{2},\\ \delta_{i,j} = \frac{\Delta \tau_{i,j} Da_{ag}^{i,j}}{2}, \quad \epsilon_{i,j} =  \frac{\Delta \tau_{i,j} Da_d^{i,j}}{2},$$

$$\Rightarrow w_{i, j+1} - w_{i, j} - \alpha_{i,j} \left( w_{i+1,j+1} - 2 w_{i,j+1} + w_{i-1,j+1} + w_{i+1,j} - 2 w_{i,j} + w_{i-1,j} \right)\\ + \beta_{i,j} \left( w_{i,j+1} - w_{i-1,j+1} + w_{i,j} - w_{i-1,j} \right)\\ - \gamma_{i,j} \left[ F_{i,j+1} \cdot H_1\left( F_{i,j+1} \right) - k_{diss} w_{i,j+1} H_0\left( -F_{i,j+1} \right) + F_{i,j} \cdot H_1\left( F_{i,j} \right) - k_{diss} w_{i,j} H_0\left( -F_{i,j} \right) \right]\\ + \delta_{i,j} \left( w_{i,j+1}^2 + w_{i,j}^2 \right) + \epsilon_{i,j} \left( w_{i,j+1} + w_{i,j} \right)= 0$$

$$\Rightarrow \left[ 1 + 2 \alpha_{i,j} + \beta_{i,j} + \gamma_{i,j} k_{diss} H_0\left( -F_{i,j+1} \right) + \epsilon_{i,j} \right] w_{i,j+1} - \alpha_{i,j} w_{i+1,j+1}\\ - \left( \alpha_{i,j} + \beta_{i,j} \right) w_{i-1,j+1}\\ + \left[ -1 + 2 \alpha_{i,j} + \beta_{i,j} + \gamma_{i,j} k_{diss} H_0\left( -F_{i,j} \right) + \epsilon_{i,j} \right] w_{i,j} - \alpha_{i,j} w_{i+1,j}\\ - \left( \alpha_{i,j} + \beta_{i,j} \right) w_{i-1,j}\\ - \gamma_{i,j} \left[ F_{i,j+1} \cdot H_1\left( F_{i,j+1} \right) + F_{i,j} \cdot H_1\left( F_{i,j} \right) \right]\\ + \delta_{i,j} \left( w_{i,j+1}^2 + w_{i,j}^2 \right) = 0$$