# Asymmetric Ternary MTJ Spiking Neural Network on MNIST

## Overview

This experiment evaluates a ternary MTJ-inspired spiking neural network for MNIST classification.

The network architecture is:

```text
784 -> 128 -> 64 -> 10
```

The model uses 25 simulation time steps and is trained for 10 epochs using Adam and cross-entropy loss.

Unlike the original binary MTJ neuron, the neuron in this experiment supports ternary signed activity:

```text
+1   positive spike
 0   no spike
-1   negative spike
```

The positive and negative output spike thresholds are asymmetric.

---

## Ternary MTJ Neuron

### Input pulse

The physical MTJ pulse is determined only by the sign of the neural input \(z\):

\[
p(z)=
\begin{cases}
+1, & z>0 \\
0, & z=0 \\
-1, & z<0
\end{cases}
\]

There is no additional pulse threshold.

The magnitude of the physical current density is determined from the magnitude of the neural input:

\[
|z| \rightarrow J
\]

Therefore, positive and negative inputs with the same magnitude generate the same current magnitude but opposite pulse directions.

---

## Signed MTJ State

The MTJ state is allowed to evolve over the signed range

\[
m_z \in [-1,1].
\]

For a nonzero input pulse, the MTJ state is updated using signed integration:

\[
m_{t+1}
=
A(J)\tanh\left[
p
\frac{t_p}{\tau_r(J)}
+
\tanh^{-1}
\left(
\frac{m_t}{A(J)}
\right)
\right],
\]

where

\[
p\in\{-1,+1\}.
\]

Thus:

- \(p=+1\) drives the MTJ state toward the positive direction.
- \(p=-1\) drives the MTJ state toward the negative direction.
- \(p=0\) applies the MTJ leakage dynamics.

The state is constrained numerically to

\[
-0.999 \le m_z \le 0.999.
\]

---

## Asymmetric Ternary Spike Emission

The output neuron uses different thresholds for positive and negative spike emission.

The spike rule is

\[
s(m_z)=
\begin{cases}
+1, & m_z > \theta_{+} \\
0, & -\theta_{-}\le m_z\le \theta_{+} \\
-1, & m_z < -\theta_{-}.
\end{cases}
\]

The implemented thresholds are:

```text
Positive spike threshold: 0.80
Negative spike threshold: 0.90
```

Therefore, a stronger negative MTJ state is required to emit a negative spike.

The network propagates ternary spikes between layers:

\[
s\in\{-1,0,+1\}.
\]

---

## Network Architecture

```text
Input image
   |
   v
Flatten: 28 x 28 -> 784
   |
   v
Linear: 784 -> 128
   |
   v
Ternary MTJ neuron
   |
   v
Linear: 128 -> 64
   |
   v
Ternary MTJ neuron
   |
   v
Linear: 64 -> 10
   |
   v
Ternary MTJ output neuron
   |
   v
Signed spike-count classification
```

Total number of trainable parameters:

```text
109,386
```

---

## Training Configuration

| Parameter | Value |
|---|---:|
| Dataset | MNIST |
| Batch size | 64 |
| Learning rate | \(10^{-3}\) |
| Optimizer | Adam |
| Loss | Cross-entropy |
| Epochs | 10 |
| Simulation steps | 25 |
| Positive spike threshold | 0.80 |
| Negative spike threshold | 0.90 |
| Pulse width | 30 ps |
| Leak interval | 100 ps |
| Leak time constant | 503.8 ps |
| Minimum current density | \(10^{11}\) |
| Maximum current density | \(10^{12}\) |
| Surrogate pulse slope parameter | 5.0 |
| Gradient clipping | 1.0 |
| Device | CUDA GPU |

---

## Spike Firing Rate

Because the neuron emits both positive and negative spikes, firing rate is calculated using the absolute value of the spike:

\[
\mathrm{FR}
=
\frac{1}{N}
\sum_i |s_i|.
\]

Thus both

\[
+1
\]

and

\[
-1
\]

are counted as firing events.

For example,

\[
[+1,-1,0,+1]
\]

has firing rate

\[
\frac{1+1+0+1}{4}=0.75.
\]

This prevents positive and negative spikes from cancelling in the activity statistic.

---

## Training Results

| Epoch | Train Loss | Train Accuracy | Test Accuracy |
|---:|---:|---:|---:|
| 1 | 0.8415 | 82.06% | 89.62% |
| 2 | 0.5060 | 90.69% | 92.00% |
| 3 | 0.4050 | 92.46% | 93.71% |
| 4 | 0.3631 | 93.72% | 94.07% |
| 5 | 0.3317 | 94.39% | 94.42% |
| 6 | 0.3042 | 95.05% | 94.77% |
| 7 | 0.2881 | 95.33% | 94.60% |
| 8 | 0.2708 | 95.67% | 95.51% |
| 9 | 0.2495 | 95.89% | **95.89%** |
| 10 | 0.2469 | 96.16% | 95.22% |

The best test accuracy was obtained at epoch 9:

\[
\boxed{95.89\%}
\]

The best model was saved and reloaded for final evaluation.

---

## Best / Final Result

```text
Best Test Accuracy: 95.89%
Final Test Accuracy: 95.89%
```

The final evaluation reproduces the best checkpoint accuracy.

---

## Final Output Spike Firing Rates

| Layer | Firing Rate |
|---|---:|
| Layer 1 | 0.9316 |
| Layer 2 | 0.8826 |
| Layer 3 | 0.7771 |

The first hidden layer has the highest spike activity, while activity decreases toward the output layer.

In percentage form:

```text
Layer 1: 93.16%
Layer 2: 88.26%
Layer 3: 77.71%
```

---

## Final Physical Pulse Rates

| Layer | Pulse Rate |
|---|---:|
| Layer 1 | 1.0000 |
| Layer 2 | 1.0000 |
| Layer 3 | 1.0000 |

All three layers have a physical pulse rate of

\[
1.0.
\]

This follows directly from the sign-based ternary pulse rule:

\[
z>0\rightarrow+1,\qquad
z<0\rightarrow-1.
\]

Since floating-point pre-activations are almost never exactly zero, nearly every neuron receives either a positive or a negative physical pulse at every simulation step.

Therefore, in this model:

```text
Physical pulse activity ~= 100%
```

while the output spike rate remains below 100% because the MTJ state still has to cross the positive or negative output threshold before a spike is emitted.

---

## Accuracy Progression

Test accuracy improved as follows:

```text
Epoch  1: 89.62%
Epoch  2: 92.00%
Epoch  3: 93.71%
Epoch  4: 94.07%
Epoch  5: 94.42%
Epoch  6: 94.77%
Epoch  7: 94.60%
Epoch  8: 95.51%
Epoch  9: 95.89%  <- best
Epoch 10: 95.22%
```

The network reaches approximately 94% test accuracy by epoch 4 and peaks at 95.89% at epoch 9.

---

## Summary

The final model is a fully ternary MTJ-inspired spiking MLP with:

- signed physical MTJ pulses \(\{-1,0,+1\}\),
- signed MTJ state \(m_z\in[-1,1]\),
- asymmetric positive and negative output spike thresholds,
- ternary inter-layer communication,
- signed spike-count classification,
- firing-rate calculation based on absolute spike activity.

The final model achieved:

\[
\boxed{\text{MNIST Test Accuracy} = 95.89\%}
\]

with final firing rates:

\[
\boxed{
FR_1=0.9316,\quad
FR_2=0.8826,\quad
FR_3=0.7771
}
\]

and physical pulse rates of 1.0 in all layers.

The 100% pulse rate is a direct consequence of using the sign of \(z\) without a dead zone: virtually every nonzero pre-activation generates either a positive or negative physical MTJ pulse.
