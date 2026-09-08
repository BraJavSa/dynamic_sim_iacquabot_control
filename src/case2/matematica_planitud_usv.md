# Formulación Matemática Completa: Planitud Diferencial y Optimización de Trayectorias para el USV Iacquabot

Este documento detalla la formulación matemática integral implementada en el módulo `optimal4`, cubriendo el modelo dinámico subactuado Fossen de 5 parámetros, la demostración de planitud diferencial, la optimización por tramos *minimum-jerk*, la solución analítica a la singularidad de frenado por inversión de $a_u$, y el mapeo de empuje a propulsores.

---

## 1. Modelo Dinámico del USV (Fossen 3-DOF, 5 Parámetros)

Basado en *Fang et al. (2024, Symmetry 16, 1118)*, el modelo cinemático y dinámico del USV en el plano horizontal (3 grados de libertad: *surge* $x$, *sway* $y$, *yaw* $\psi$) se define como:

### 1.1 Cinemática (Transformación al Marco Inercial)
$$
\begin{aligned}
\dot{x} &= u \cos\psi - v \sin\psi \\
\dot{y} &= u \sin\psi + v \cos\psi \\
\dot{\psi} &= r
\end{aligned}
$$

Matricialmente:
$$
\dot{\mathbf{F}} = \begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix} = \mathbf{R}(\psi) \begin{bmatrix} u \\ v \end{bmatrix}, \quad \mathbf{R}(\psi) = \begin{bmatrix} \cos\psi & -\sin\psi \\ \sin\psi & \cos\psi \end{bmatrix}
$$

### 1.2 Dinámica en el Marco del Cuerpo (*Body Frame*)
$$
\begin{aligned}
\dot{u} &= \frac{X_u}{m} u + v r + \frac{1}{m} \tau_u = \beta_1 u + v r + \tau_1 \\
\dot{v} &= \frac{Y_v}{m} v - u r = \beta_2 v - u r \quad (\text{Sin fuerza lateral } \tau_v = 0) \\
\dot{r} &= \frac{N_r}{I_z} r + \frac{1}{I_z} \tau_r = \beta_3 r + \tau_3
\end{aligned}
$$

donde las variables normalizadas y coeficientes son:
* $\beta_1 = \frac{X_u}{m}$ (amortiguamiento lineal de surge normalizado)
* $\beta_2 = \frac{Y_v}{m}$ (amortiguamiento lineal de sway normalizado)
* $\beta_3 = \frac{N_r}{I_z}$ (amortiguamiento lineal de yaw normalizado)
* $\tau_1 = \frac{\tau_u}{m}$ (fuerza de surge específica, $\text{N/kg}$)
* $\tau_3 = \frac{\tau_r}{I_z}$ (aceleración angular específica de yaw, $\text{rad/s}^2$)

> **Nota de Subactuación**: La ecuación de sway no tiene término de control directo ($\tau_v = 0$). El sway $v$ se acopla pasivamente a través de $\dot{v} = \beta_2 v - u r$.

---

## 2. Demostración de Planitud Diferencial (Proposición 1)

Un sistema es **diferencialmente plano** si existe un conjunto de salidas $\mathbf{F}(t)$ (igual en número a las entradas de control actuadas) tal que todos los estados $(x, y, \psi, u, v, r)$ y las entradas de control $(\tau_u, \tau_r)$ pueden expresarse como funciones algebraicas de $\mathbf{F}(t)$ y sus derivadas finitas en el tiempo.

### 2.1 Elección de la Salida Plana
Elegimos la posición inercial en el plano 2D como única salida plana:
$$
\mathbf{F}(t) = \begin{bmatrix} x(t) \\ y(t) \end{bmatrix} \in \mathbb{R}^2
$$

### 2.2 Reconstrucción Cinemática
Despejando las velocidades del marco cuerpo de la derivada primera $\dot{\mathbf{F}} = \mathbf{R}(\psi) [u, v]^T$:
$$
\begin{bmatrix} u \\ v \end{bmatrix} = \mathbf{R}^T(\psi) \begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix} = \begin{bmatrix} \dot{x} \cos\psi + \dot{y} \sin\psi \\ -\dot{x} \sin\psi + \dot{y} \cos\psi \end{bmatrix}
$$

Derivando $\dot{\mathbf{F}}$ con respecto al tiempo:
$$
\ddot{\mathbf{F}} = \dot{\mathbf{R}}(\psi) \begin{bmatrix} u \\ v \end{bmatrix} + \mathbf{R}(\psi) \begin{bmatrix} \dot{u} \\ \dot{v} \end{bmatrix}
$$
Como $\dot{\mathbf{R}}(\psi) = \mathbf{R}(\psi) \mathbf{S}(r)$ con $\mathbf{S}(r) = \begin{bmatrix} 0 & -r \\ r & 0 \end{bmatrix}$:
$$
\ddot{\mathbf{F}} = \mathbf{R}(\psi) \left( \begin{bmatrix} 0 & -r \\ r & 0 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} + \begin{bmatrix} \dot{u} \\ \dot{v} \end{bmatrix} \right) = \mathbf{R}(\psi) \begin{bmatrix} \dot{u} - v r \\ \dot{v} + u r \end{bmatrix}
$$

Sustituyendo la ecuación pasiva de sway $\dot{v} + u r = \beta_2 v$:
$$
\ddot{\mathbf{F}} = \mathbf{R}(\psi) \begin{bmatrix} \dot{u} - v r \\ \beta_2 v \end{bmatrix}
$$

Ahora restamos $\beta_2 \dot{\mathbf{F}} = \beta_2 \mathbf{R}(\psi) \begin{bmatrix} u \\ v \end{bmatrix}$:
$$
\ddot{\mathbf{F}} - \beta_2 \dot{\mathbf{F}} = \mathbf{R}(\psi) \begin{bmatrix} \dot{u} - v r - \beta_2 u \\ 0 \end{bmatrix}
$$

Definiendo el escalar de aceleración efectiva de surge $a_u(t) \equiv \dot{u} - v r - \beta_2 u$:
$$
\begin{bmatrix} \ddot{x} - \beta_2 \dot{x} \\ \ddot{y} - \beta_2 \dot{y} \end{bmatrix} = a_u \begin{bmatrix} \cos\psi \\ \sin\psi \end{bmatrix}
$$

Esta es la **relación fundamental de la planitud diferencial del USV subactuado**.

---

## 3. Análisis de la Singularidad de Frenado y Solución Analítica

### 3.1 El Origen del Problema ($\text{atan2}$ Ciego)
En la literatura (*Fang et al. 2024*), se propone despejar $\psi$ dividiendo la componente $y$ entre la $x$:
$$
\tan\psi = \frac{\ddot{y} - \beta_2 \dot{y}}{\ddot{x} - \beta_2 \dot{x}} \implies \psi_{naive} = \operatorname{atan2}(\ddot{y} - \beta_2 \dot{y}, \,\, \ddot{x} - \beta_2 \dot{x})
$$

Tomando la norma euclidiana del vector:
$$
\sqrt{(\ddot{x} - \beta_2 \dot{x})^2 + (\ddot{y} - \beta_2 \dot{y})^2} = |a_u|
$$

Si $a_u > 0$ (aceleración o avance constante), el vector $\ddot{\mathbf{F}} - \beta_2 \dot{\mathbf{F}}$ apunta en la misma dirección que la orientación $\psi$.

**Sin embargo, durante el frenado o desaceleración al aproximarse a un waypoint:**
* La aceleración $\dot{u}$ es fuertemente negativa ($\dot{u} < 0$).
* Como $\beta_2 = \frac{Y_v}{m} < 0$, el término $-\beta_2 u$ es positivo pero menor en magnitud que $|\dot{u}|$.
* En consecuencia, **$a_u = \dot{u} - v r - \beta_2 u$ cambia de signo y se vuelve NEGATIVO ($a_u < 0$)**.

Cuando $a_u < 0$:
$$
\begin{bmatrix} \ddot{x} - \beta_2 \dot{x} \\ \ddot{y} - \beta_2 \dot{y} \end{bmatrix} = -|a_u| \begin{bmatrix} \cos\psi \\ \sin\psi \end{bmatrix} = |a_u| \begin{bmatrix} \cos(\psi \pm \pi) \\ \sin(\psi \pm \pi) \end{bmatrix}
$$

Evaluar `atan2` ciegamente en esta condición devuelve $\psi \pm \pi$ ($180^\circ$ opuesto al rumbo real).

### 3.2 Prueba Matemática del Cruce por Cero en Trayectorias *Minimum-Jerk*
Para cualquier tramo polinomial de grado 5 con condiciones de reposo a reposo ($\dot{x}(0)=0, \ddot{x}(0)=0, \dot{x}(T)=0, \ddot{x}(T)=0$):

$$
\text{den}(s) = \frac{L_x}{T_{seg}^2} 30 s (1 - s) \left[ 2 (1 - 2 s) + 0.6237 T_{seg} s (1 - s) \right], \quad s = \frac{t}{T_{seg}} \in [0, 1]
$$

Definiendo $Q(s) = 2 (1 - 2 s) + 0.6237 T_{seg} s (1 - s)$:
* En el inicio del tramo ($s \to 0^+$): $Q(0) = +2 > 0 \implies \text{den} > 0$.
* En el final del tramo ($s \to 1^-$): $Q(1) = -2 < 0 \implies \text{den} < 0$.

Por el **Teorema del Valor Intermedio**, $Q(s)$ tiene al menos una raíz en $s^* \in (0, 1)$. En ese punto, el vector de fuerzas cruza por cero y se invierte. 

Forzar $\psi = \operatorname{atan2}(\text{num}, \text{den})$ fuerza al USV a dar un **giro artificial instantáneo de $180^\circ$** sobre su eje para frenar en lugar de mantener su rumbo y aplicar empuje en reversa ($\tau_u < 0$).

### 3.3 Reconstrucción de Rumbo Continuo y Consistente
Para una navegación en avance ($u > 0$), la orientación física del barco $\psi(t)$ es continua y sigue la dirección del vector velocidad de avance $\mathbf{v}(t) = [\dot{x}(t), \dot{y}(t)]^T$:

$$
\psi(t) = \operatorname{unwrap} \left( \operatorname{atan2}(\dot{y}(t), \dot{x}(t)) \right)
$$

En las regiones de velocidad nula ($\|\dot{\mathbf{F}}\| < 10^{-3} \text{ m/s}$), $\psi(t)$ se interpola suavemente desde las muestras contiguas válidas.

---

## 4. Reconstrucción de Fuerzas Generalizadas Actuadas ($\tau_u, \tau_r$)

Con la trayectoria de posición $\mathbf{F}(t) = [x(t), y(t)]^T$ y el rumbo continuo $\psi(t)$:

1. **Velocidades del marco cuerpo**:
   $$
   u(t) = \dot{y} \sin\psi + \dot{x} \cos\psi
   $$
   $$
   v(t) = \dot{y} \cos\psi - \dot{x} \sin\psi
   $$
   $$
   r(t) = \dot{\psi} = \frac{d\psi}{dt}
   $$

2. **Fuerza de Surge ($\tau_u$)**:
   Proyectando las ecuaciones dinámicas del paper sobre el rumbo $\psi$:
   $$
   \tau_1 = (\ddot{y} - \beta_1 \dot{y}) \sin\psi + (\ddot{x} - \beta_1 \dot{x}) \cos\psi
   $$
   $$
   \tau_u = m \cdot \tau_1 \quad [\text{N}]
   $$
   *(Nota: Cuando el barco desacelera, esta ecuación calcula naturalmente $\tau_u < 0$, representando empuje en reversa sin alterar $\psi$)*.

3. **Torque de Yaw ($\tau_r$)**:
   $$
   \tau_3 = \dot{r} + \beta_3 r = \ddot{\psi} + \beta_3 \dot{\psi}
   $$
   $$
   \tau_r = I_z \cdot \tau_3 \quad [\text{N}\cdot\text{m}]
   $$

---

## 5. Generación de Trayectoria Minimum-Jerk (QP por Tramos)

Para minimizar sacudidas (*jerk*) en el plano inercial $x(t), y(t)$:

$$
\min_{x(t), y(t)} \int_{0}^{T} \left( \left(\frac{d^3 x}{dt^3}\right)^2 + \left(\frac{d^3 y}{dt^3}\right)^2 \right) dt
$$

### 5.1 Parametrización Polinomial
Cada eje se resuelve independientemente mediante polinomios de grado 5 por tramo $k \in \{1, \dots, K\}$:
$$
p_k(\tau) = \sum_{i=0}^{5} c_{k,i} \tau^i, \quad \tau = t - t_k \in [0, T_k]
$$

### 5.2 Formulación del Problema Cuadrático (QP) y KKT
Vector de coeficientes $\mathbf{c} \in \mathbb{R}^{6K}$:

$$
\min_{\mathbf{c}} \frac{1}{2} \mathbf{c}^T \mathbf{H} \mathbf{c} \quad \text{s.t.} \quad \mathbf{A} \mathbf{c} = \mathbf{b}
$$

Matriz de costo en bloque diagonal $\mathbf{H}_k \in \mathbb{R}^{6 \times 6}$:
$$
H_{k, i, j} = \int_{0}^{T_k} \frac{d^3 (\tau^i)}{d\tau^3} \frac{d^3 (\tau^j)}{d\tau^3} d\tau = \frac{i(i-1)(i-2) \cdot j(j-1)(j-2)}{(i-3) + (j-3) + 1} T_k^{(i-3)+(j-3)+1}, \quad i,j \ge 3
$$

Restricciones de igualdad $\mathbf{A} \mathbf{c} = \mathbf{b}$:
1. **Paso por Waypoints**: $p_k(0) = w_k$, $p_k(T_k) = w_{k+1}$.
2. **Continuidad $C^2$ en nodos internos**: $\dot{p}_k(T_k) = \dot{p}_{k+1}(0)$, $\ddot{p}_k(T_k) = \ddot{p}_{k+1}(0)$.
3. **Condiciones de reposo en extremos**: $\dot{p}_1(0) = \ddot{p}_1(0) = 0$, $\dot{p}_K(T_K) = \ddot{p}_K(T_K) = 0$.

El sistema lineal KKT exacto a resolver es:
$$
\begin{bmatrix} \mathbf{H} & \mathbf{A}^T \\ \mathbf{A} & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{c} \\ \boldsymbol{\lambda} \end{bmatrix} = \begin{bmatrix} \mathbf{0} \\ \mathbf{b} \end{bmatrix}
$$

---

## 6. Asignación de Empuje y Mapeo No Lineal a Propulsores

### 6.1 Matriz de Asignación de Actuadores
El USV dispone de 4 propulsores en configuración en X. Dado que no existen propulsores laterales puros, la matriz de configuración recortada $B_{act} \in \mathbb{R}^{2 \times 4}$ relaciona el vector actuado $\boldsymbol{\tau} = [\tau_u, \tau_r]^T$ con las fuerzas individuales $\mathbf{T} = [T_{FR}, T_{FL}, T_{BR}, T_{BL}]^T$:

$$
\boldsymbol{\tau} = B_{act} \mathbf{T} = \begin{bmatrix} 1.0 & 1.0 & 1.0 & 1.0 \\ 1.027135 & -1.027135 & 1.027135 & -1.027135 \end{bmatrix} \begin{bmatrix} T_{FR} \\ T_{FL} \\ T_{BR} \\ T_{BL} \end{bmatrix}
$$

La inversión óptima mediante pseudo-inversa de Moore-Penrose $B_{act}^\dagger = B_{act}^T (B_{act} B_{act}^T)^{-1} \in \mathbb{R}^{4 \times 2}$:
$$
\mathbf{T}(t) = \boldsymbol{\tau}(t) \cdot (B_{act}^\dagger)^T
$$

### 6.2 Curva de Propulsión Inversa
Cada empuje deseado $T_i$ se convierte a un comando normalizado $c_i \in [-1, 1]$ resolviendo el polinomio de grado 5 ajustado a la sigmoide física identificada:

$$
P_5(c_i) - T_i = 0 \implies c_i = \operatorname{clip}(\text{root\_real}(P_5(c) = T_i), \, -1.0, \, 1.0)
$$

---

## Resumen de Resultados Numéricos

| Métrica | Enfoque Original (`atan2` ciego) | Enfoque Corregido (Rumbo Continuo) |
| :--- | :--- | :--- |
| **Max Empuje Demandado ($T_{max}$)** | $106,849.61 \text{ N}$ | **$32.34 \text{ N}$** (Límite: $36.38 \text{ N}$) |
| **Min Empuje Demandado ($T_{min}$)** | $-106,849.09 \text{ N}$ | **$-22.13 \text{ N}$** (Límite: $-28.44 \text{ N}$) |
| **Velocidad Angular Max ($|r|$)** | $10.48 \text{ rad/s}$ | **$0.34 \text{ rad/s}$** ($19.5^\circ/\text{s}$) |
| **Aceleración Angular Max ($|\dot{r}|$)** | $1322.66 \text{ rad/s}^2$ | **$0.30 \text{ rad/s}^2$** |
| **Muestras en Saturación** | Violación total / Aborto | **0 de 6628 (0.0%)** |
| **Tiempo Total de Maniobra** | No convergía (Fail at 339s) | **$55.20 \text{ s}$** (Convergencia en Iter 1) |
