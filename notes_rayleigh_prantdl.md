# Variables to change

Ra = 1e4;          // Rayleigh number  
Pr = 0.71;         // Prandtl number  
Thot = 274.15;     // temperature of the lower wall in Kelvin  
Tcold = 273.15;    // temperature of the fluid in Kelvin  


# Text matrix full range

The range of the Prantdl number is based on [table](../artigos/values-range/prantdl_table_9.3.pdf) and [range](../artigos/values-range/prantdl_range_sec.-5.3.2.pdf). The table from [table](../artigos/values-range/prantdl_table_9.3.pdf) defines a range from $0.01$ to $7612.74$ while from [range](../artigos/values-range/prantdl_range_sec.-5.3.2.pdf) mentions from $10^{-4}$ to $10^3$. The *table* varies from metal to different oils. I decided to fix the lower bound with the value around the air ($0.72$) with $0.5$ and the upper limit the refernce from [range](../artigos/values-range/prantdl_range_sec.-5.3.2.pdf), which is $10000$.


The Rayleigh range according to [rayleigh](../artigos/values-range/rayleigh_range.pdf) can vary from less than 1 and reach a critical number for turbulent induced by a heated plate with $2\cdot10^7$. I decided to set the upper bound with $10^8$.


# Text matrix reduced range

Comparative with paper from [nature](../artigos/values-range/rayleigh_benard_nature.pdf), $2.3\cdot 10^3 < Ra < 10^7$ and $5\cdot10^{-4} < Pr < 70$. The range used in this work was $10^3 < Ra < 10^7$ and $10^{-1} < Pr < 70$

The temperature range was set to be $0 °C < T_{hot} < 90 °C$ and the $1°C <$ $\Delta T$ $< 10°C$

Issues with Ra as higher as 10^7