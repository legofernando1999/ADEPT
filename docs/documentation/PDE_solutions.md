$$\tau = \frac{L}{U_z}t, \, Z = \frac{z}{L}, \, Pe = \frac{U_z L}{D_{ax}}, \, Da = \frac{k L}{U_z}$$

PDE with dimensionless variables:
$$\frac{\partial C}{\partial \tau} = \frac{1}{Pe} \frac{\partial^2 C}{\partial Z^2} - \frac{\partial C}{\partial Z} + Da_p r_p - Da_ag C^2 - Da_d C$$

where:
$$r_p = \begin{cases}
        C_f - C_{eq}, &  C_f \geq C_{eq}\\
        -k_{diss} C, & C_f < C_{eq}
    \end{cases}$$

Now let's use the finite difference method to solve this PDE

**Forward-difference method, jth step in time:** 
$$\frac{w_{i, j+1} - w_{i, j}}{\Delta \tau _{i,j}} = \frac{1}{Pe^{i,j}} \left( \frac{w_{i+1,j} - 2 w_{i,j} + w_{i-1,j}}{(\Delta Z)^2} \right) -  \left( \frac{w_{i,j} - w_{i-1,j}}{ \Delta Z} \right) + Da_p^{i,j} r_p^{i,j} - Da_{ag}^{i,j} w_{i,j}^2 - Da_d^{i,j} w_{i,j}$$

**Backward-difference method, (j+1)st step in time:** 
$$\frac{w_{i, j+1} - w_{i, j}}{\Delta \tau _{i,j+1}} = \frac{1}{Pe^{i,j+1}} \left( \frac{w_{i+1,j+1} - 2 w_{i,j+1} + w_{i-1,j+1}}{(\Delta Z)^2} \right) -  \left( \frac{w_{i,j+1} - w_{i-1,j+1}}{ \Delta Z} \right) + Da_p^{i,j+1} r_p^{i,j+1} - Da_{ag}^{i,j+1} w_{i,j+1}^2 - Da_d^{i,j+1} w_{i,j+1}$$

**Crank-Nicolson (averaged) method:**

Since we only have information from the jth step in time, we will assume that $\Delta \tau _{i,j+1}\approx \Delta \tau _{i,j}$, $Pe^{i,j+1} \approx Pe^{i,j}$, $Da_p^{i,j+1} \approx Da_p^{i,j}$, $Da_{ag}^{i,j+1} \approx Da_{ag}^{i,j}$ and $Da_d^{i,j+1} \approx Da_d^{i,j}$

$$\frac{w_{i, j+1} - w_{i, j}}{\Delta \tau _{i,j}} = \frac{1}{2 Pe^{i,j}} \left( \frac{w_{i+1,j+1} - 2 w_{i,j+1} + w_{i-1,j+1}}{(\Delta Z)^2} + \frac{w_{i+1,j} - 2 w_{i,j} + w_{i-1,j}}{(\Delta Z)^2} \right)\\ - \frac{1}{2}\left( \frac{w_{i,j+1} - w_{i-1,j+1}}{ \Delta Z}+ \frac{w_{i,j} - w_{i-1,j}}{ \Delta Z} \right) + \frac{Da_p^{i,j}}{2} \left( r_p^{i,j+1} + r_p^{i,j} \right) - \frac{Da_{ag}^{i,j}}{2} \left( w_{i,j+1}^2 + w_{i,j}^2 \right) - \frac{Da_d^{i,j}}{2} \left( w_{i,j+1} + w_{i,j} \right)$$

$$w_{i, j+1} - w_{i, j} = \alpha_{i,j} \left( w_{i+1,j+1} - 2 w_{i,j+1} + w_{i-1,j+1} + w_{i+1,j} - 2 w_{i,j} + w_{i-1,j} \right) - \beta_{i,j} \left( w_{i,j+1} - w_{i-1,j+1} + w_{i,j} - w_{i-1,j} \right)\\ + \gamma_{i,j} \left( r_p^{i,j+1} + r_p^{i,j} \right) - \delta_{i,j} \left( w_{i,j+1}^2 + w_{i,j}^2 \right) - \epsilon_{i,j} \left( w_{i,j+1} + w_{i,j} \right)$$

Where:

$$\alpha_{i,j} = \frac{\Delta \tau _{i,j}}{2 Pe^{i,j} (\Delta Z)^2}, \quad \beta_{i,j} = \frac{\Delta \tau _{i,j}}{2 \Delta Z}, \quad \gamma_{i,j} =  \frac{\Delta \tau _{i,j} Da_p^{i,j}}{2}, \quad \delta_{i,j} = \frac{\Delta \tau _{i,j} Da_{ag}^{i,j}}{2}, \quad \epsilon_{i,j} = \frac{\Delta \tau _{i,j} Da_d^{i,j}}{2}$$

$$\Rightarrow w_{i, j+1} - w_{i, j} - \alpha_{i,j} \left( w_{i+1,j+1} - 2 w_{i,j+1} + w_{i-1,j+1} + w_{i+1,j} - 2 w_{i,j} + w_{i-1,j} \right)\\ + \beta_{i,j} \left( w_{i,j+1} - w_{i-1,j+1} + w_{i,j} - w_{i-1,j} \right) - \gamma_{i,j} \left( r_p^{i,j+1} + r_p^{i,j} \right)\\ + \delta_{i,j} \left( w_{i,j+1}^2 + w_{i,j}^2 \right) + \epsilon_{i,j} \left( w_{i,j+1} + w_{i,j} \right) = 0$$

$$\Rightarrow \left( 1 + 2 \alpha_{i,j} + \beta_{i,j} + \epsilon_{i,j} \right) w_{i,j+1} - \alpha_{i,j} w_{i+1,j+1} - \left( \alpha_{i,j} + \beta_{i,j} \right) w_{i-1,j+1}\\ + \left( -1 + 2 \alpha_{i,j} + \beta_{i,j} + \epsilon_{i,j} \right) w_{i,j} - \alpha_{i,j} w_{i+1,j} - \left( \alpha_{i,j} + \beta_{i,j} \right) w_{i-1,j}\\ - \gamma_{i,j} \left( r_p^{i,j+1} + r_p^{i,j} \right) + \delta_{i,j} \left( w_{i,j+1}^2 + w_{i,j}^2 \right) = 0$$